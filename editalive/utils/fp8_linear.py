import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger()


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def quant_fp8(input, target_dtype=torch.float8_e4m3fn):
    max_value = torch.finfo(target_dtype).max
    amax_input = torch.max(torch.abs(input)).float()
    input_scale = (max_value / torch.clamp(amax_input, min=1e-12)).clamp(max=max_value)
    input_fp8 = (input * input_scale).clamp(-max_value, max_value).to(target_dtype)
    return input_fp8, input_scale.reciprocal()


class FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, max_input_val, max_weight_val):
        """
        input: [B, *, in_features]  (可以是 2D 或 3D)
        weight: [out_features, in_features]
        bias: [out_features] 或 None
        """
        prev_shape = input.shape  # 保存原始形状

        # ===== 权重量化 =====
        if isinstance(weight, tuple):
            input_2d = input.view(-1, weight[0].shape[1])
            weight_fp8, weight_scale = weight
            out_feature = weight[0].shape[0]
        else:
            input_2d = input.view(-1, weight.shape[1])
            weight_fp8, weight_scale = quant_fp8(weight)
            out_feature = weight.shape[0]

        # ===== 输入量化 (PerTensor) =====
        input_fp8, input_scale = quant_fp8(input_2d, torch.float8_e4m3fn)

        # ===== FP8 matmul =====
        out_2d = torch._scaled_mm(
            input_fp8,
            weight_fp8.T,
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        # 恢复成原来的 batch/seq 形状
        if isinstance(out_2d, tuple):
            out_2d = out_2d[0]
        # Keep _scaled_mm's bf16 output, as nn.Linear under the transformer's bf16
        # autocast would. Casting back to input.dtype made every layer fed from
        # the fp32 residual stream emit fp32, which also stored the rolling KV
        # cache in fp32 (twice its bf16 size).
        return out_2d.view(*prev_shape[:-1], out_feature)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("no implement for backward")


class FP8ScaleLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16, device="cuda"):
        super().__init__()
        # Kept like nn.Linear's: callers (e.g. the pipeline warmup) read them.
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.bias = None
        self.reset_parameters()
        self.max_input_val = torch.finfo(torch.float8_e4m3fn).max
        self.max_weight_val = torch.finfo(torch.float8_e4m3fn).max
        self.quantized_weight = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weight(self,):
        fp8_data, scale = quant_fp8(self.weight.detach())
        self.register_buffer("weight_fp8", fp8_data)
        self.register_buffer("weight_scale", scale)
        meta_w = torch.empty(self.weight.shape, device='meta', dtype=self.weight.dtype)
        self._parameters.pop('weight')
        self.weight = meta_w
        self.quantized_weight = True

    @classmethod
    def from_linear(cls, linear: nn.Linear, quantize_weight=True):
        new_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            dtype=linear.weight.dtype
        )
        new_layer = new_layer.to(linear.weight.device)
        with torch.no_grad():
            new_layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_layer.bias.copy_(linear.bias)
        if quantize_weight:
            new_layer.quantize_weight()
        return new_layer

    def forward(self, input):
        if self.quantized_weight:
            W_eff = (self.weight_fp8, self.weight_scale)
        else:
            W_eff = self.weight
        return FP8LinearFunction.apply(
            input, W_eff, self.bias, self.max_input_val, self.max_weight_val)


def contains_substring(str_list, target_str):
    for s in str_list:
        if s in target_str:
            return True
    return False


def mark_fp8_ignored(module: nn.Module, ignore_keys):
    """Tag layers to keep in bf16, by full name, before converting block by block."""
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and contains_substring(ignore_keys, name):
            setattr(child, "need_fp8", False)


def replace_linear_with_scaled_fp8(module: nn.Module, ignore_keys=[], quantize_weight=True):
    if len(ignore_keys) > 0:
        mark_fp8_ignored(module, ignore_keys)

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not hasattr(child, "need_fp8"):
            setattr(module, name, FP8ScaleLinear.from_linear(child, quantize_weight=quantize_weight))
        else:
            replace_linear_with_scaled_fp8(child, quantize_weight=quantize_weight)


__all__ = ['FP8ScaleLinear', 'replace_linear_with_scaled_fp8', 'mark_fp8_ignored',
           'quant_fp8']
