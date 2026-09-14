import torch
from safetensors.torch import load_file


def _load_lora_state_dict(
    lora_path,
    state_dict=None,
    log_prefix="[LoRA Merge]",
    verbose=True,
):
    if state_dict is not None:
        if verbose:
            print(f"{log_prefix} Using provided state_dict")
        return state_dict

    if lora_path is None:
        raise ValueError("lora_path must be provided when state_dict is None")

    if lora_path.endswith("safetensors"):
        if verbose:
            print(f"{log_prefix} Loading safetensors file...")
        return load_file(lora_path)

    if verbose:
        print(f"{log_prefix} Loading pytorch file...")
    return torch.load(lora_path, map_location="cpu")

def _strip_lora_known_prefix(name):
    prefixes = [
        "base_model.model.diffusion_model.",
        "base_model.model.transformer.",
        "base_model.model.",
        "diffusion_model.",
        "model.diffusion_model.",
        "transformer.",
        "unet.",
        "model.",
        "module.",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name

def _collect_lora_groups(lora_state):
    groups = {}
    suffix_targets = [
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".alpha", "alpha"),
        ("_lora_down_weight", "down"),
        ("_lora_up_weight", "up"),
        ("_lora_A_default_weight", "down"),
        ("_lora_B_default_weight", "up"),
        ("_lora_A_weight", "down"),
        ("_lora_B_weight", "up"),
        ("_alpha", "alpha"),
    ]

    for key in lora_state:
        clean_key = _strip_lora_known_prefix(key)
        if ".lora_A." in clean_key and clean_key.endswith(".weight"):
            module_key = clean_key.split(".lora_A.")[0]
            groups.setdefault(module_key, {})["down"] = key
        elif ".lora_B." in clean_key and clean_key.endswith(".weight"):
            module_key = clean_key.split(".lora_B.")[0]
            groups.setdefault(module_key, {})["up"] = key
        else:
            for suffix, target in suffix_targets:
                if clean_key.endswith(suffix):
                    module_key = clean_key[:-len(suffix)]
                    groups.setdefault(module_key, {})[target] = key
                    break

    return groups

def _collect_diff_weight(lora_state):
    diff_weight = {}
    for key in lora_state:
        clean_key = _strip_lora_known_prefix(key)
        if clean_key.endswith(".diff"):
            diff_weight[clean_key[:-len(".diff")]] = key
        elif clean_key.endswith("_diff") and not clean_key.endswith("_diff_b"):
            diff_weight[clean_key[:-len("_diff")]] = key
    return diff_weight

def _collect_diff_bias(lora_state):
    diff_bias = {}
    for key in lora_state:
        clean_key = _strip_lora_known_prefix(key)
        if clean_key.endswith(".diff_b"):
            diff_bias[clean_key[:-len(".diff_b")]] = key
        elif clean_key.endswith("_diff_b"):
            diff_bias[clean_key[:-len("_diff_b")]] = key
    return diff_bias

def _normalize_lora_module_key(module_key):
    module_key = _strip_lora_known_prefix(module_key)
    for prefix in ("lora_unet__", "lora_unet_", "lora_unet.", "lora_te__", "lora_te_", "lora_te."):
        if module_key.startswith(prefix):
            module_key = module_key[len(prefix):]
            break
    return module_key

def _is_text_encoder_lora_key(module_key):
    module_key = _strip_lora_known_prefix(module_key)
    return module_key.startswith(("lora_te", "text_encoder"))

def _build_transformer_module_lookup(transformer):
    module_lookup = {}

    def register(alias, module):
        if alias:
            module_lookup.setdefault(alias, module)

    for name, module in transformer.named_modules():
        if not name:
            continue
        underscore_name = name.replace(".", "_")
        register(name, module)
        register(underscore_name, module)
        register("lora_unet_" + underscore_name, module)
        register("lora_unet__" + underscore_name, module)

    return module_lookup

def _find_transformer_module(module_key, module_lookup):
    clean_key = _normalize_lora_module_key(module_key)
    candidates = [
        module_key,
        clean_key,
        clean_key.replace(".", "_"),
        clean_key.replace("_", "."),
    ]

    for candidate in candidates:
        if candidate in module_lookup:
            return module_lookup[candidate]

    suffix_matches = [
        module
        for key, module in module_lookup.items()
        if key.endswith(clean_key) or key.endswith(clean_key.replace(".", "_"))
    ]
    unique_matches = []
    for module in suffix_matches:
        if not any(module is seen for seen in unique_matches):
            unique_matches.append(module)

    return unique_matches[0] if len(unique_matches) == 1 else None

def _compute_lora_delta(up, down, base_weight):
    up = up.float()
    down = down.float()

    if down.ndim == 2 and up.ndim == 2:
        delta = up @ down
        if base_weight.ndim == 2:
            return delta
        if base_weight.ndim == 4 and base_weight.shape[2:] == (1, 1):
            return delta[:, :, None, None]

    if down.ndim == 4 and up.ndim == 4:
        if down.shape[2:] == (1, 1) and up.shape[2:] == (1, 1):
            delta = up.squeeze(-1).squeeze(-1) @ down.squeeze(-1).squeeze(-1)
            if base_weight.ndim == 2:
                return delta
            if base_weight.ndim == 4:
                return delta[:, :, None, None]

        if up.shape[2:] == (1, 1) and base_weight.ndim == 4:
            up_2d = up.squeeze(-1).squeeze(-1)
            return torch.einsum("or,rijk->oijk", up_2d, down)

    raise ValueError(
        f"unsupported shapes: up={tuple(up.shape)}, down={tuple(down.shape)}, base={tuple(base_weight.shape)}"
    )

def _normalize_lora_paths_and_multipliers(lora_paths, multipliers, state_dict=None):
    if lora_paths is None or lora_paths == []:
        if state_dict is not None:
            if multipliers is None:
                return [(None, 1.0)]
            if isinstance(multipliers, (int, float)):
                return [(None, float(multipliers))]
            multipliers = list(multipliers)
            if len(multipliers) != 1:
                raise ValueError("state_dict merge supports exactly one multiplier")
            return [(None, float(multipliers[0]))]
        if multipliers is not None and multipliers != []:
            raise ValueError("LoRA multipliers were provided but lora_paths is empty")
        return []

    if isinstance(lora_paths, str):
        lora_paths = [lora_paths]
    else:
        lora_paths = list(lora_paths)

    if multipliers is None:
        multipliers = [1.0] * len(lora_paths)
    elif isinstance(multipliers, (int, float)):
        multipliers = [float(multipliers)] * len(lora_paths)
    else:
        multipliers = list(multipliers)
        # One value stands for "this strength for every LoRA".
        if len(multipliers) == 1:
            multipliers = multipliers * len(lora_paths)

    if len(lora_paths) != len(multipliers):
        raise ValueError(
            f"lora_paths and multipliers must have the same length, got {len(lora_paths)} and {len(multipliers)}"
        )

    return [(lora_path, float(multiplier)) for lora_path, multiplier in zip(lora_paths, multipliers)]

@torch.no_grad()
def plan_lora_updates(transformer, lora_paths, multipliers=None, verbose=True):
    """Resolve LoRA files into the weight updates they apply, without applying them.

    Returns ``{module: [update, ...]}`` keyed by the target module object, where
    an update is ``("lora", up, down, scale)``, ``("diff", delta, multiplier)``
    or ``("diff_b", delta, multiplier)``. Nothing is read off the weights here,
    so the transformer can still be on the CPU: the caller then moves it to the
    GPU a block at a time and calls :func:`apply_lora_updates` on each piece,
    which is what keeps the fp8 load from materialising the whole bf16 model on
    the GPU first.

    Updates are ordered by file, so applying them in list order reproduces the
    sequential merge the ``--lora_paths`` order describes.
    """
    merge_items = _normalize_lora_paths_and_multipliers(lora_paths, multipliers)
    updates = {}
    if not merge_items:
        return updates

    module_lookup = _build_transformer_module_lookup(transformer)

    def add(module, update):
        updates.setdefault(module, []).append(update)

    for lora_path, multiplier in merge_items:
        state = _load_lora_state_dict(
            lora_path, log_prefix="[LoRA]", verbose=verbose)
        skipped = []

        for module_key, item in _collect_lora_groups(state).items():
            if _is_text_encoder_lora_key(module_key):
                skipped.append((module_key, "text encoder lora skipped"))
                continue
            if "up" not in item or "down" not in item:
                skipped.append((module_key, "missing lora up/down pair"))
                continue
            target = _find_transformer_module(module_key, module_lookup)
            if target is None or not hasattr(target, "weight"):
                skipped.append((module_key, "target weight not found"))
                continue

            up = state[item["up"]]
            down = state[item["down"]]
            rank = down.shape[0]
            alpha = (float(state[item["alpha"]].float().item())
                     if "alpha" in item else float(rank))
            add(target, ("lora", up, down, multiplier * alpha / rank))

        for module_key, diff_key in _collect_diff_weight(state).items():
            if _is_text_encoder_lora_key(module_key):
                skipped.append((module_key, "text encoder lora skipped"))
                continue
            target = _find_transformer_module(module_key, module_lookup)
            if target is None or not hasattr(target, "weight"):
                skipped.append((module_key, "target weight not found"))
                continue
            add(target, ("diff", state[diff_key], multiplier))

        for module_key, diff_key in _collect_diff_bias(state).items():
            if _is_text_encoder_lora_key(module_key):
                skipped.append((module_key, "text encoder lora skipped"))
                continue
            target = _find_transformer_module(module_key, module_lookup)
            if target is None or getattr(target, "bias", None) is None:
                skipped.append((module_key, "target bias not found"))
                continue
            add(target, ("diff_b", state[diff_key], multiplier))

        if skipped:
            print(f"[LoRA] {lora_path}: skipped {len(skipped)} entries, e.g.")
            for item in skipped[:20]:
                print(f"  {item}")

    return updates


@torch.no_grad()
def apply_lora_updates(updates, piece):
    """Fuse every planned update whose target lies inside ``piece``.

    Deltas are computed in fp32 on whatever device the target weight already
    sits on, then added in the weight's own dtype. Applied updates are removed
    from ``updates``, so a module is never fused twice even when the caller
    walks overlapping pieces.
    """
    for module in piece.modules():
        for update in updates.pop(module, ()):
            kind = update[0]
            if kind == "diff_b":
                _, delta, multiplier = update
                base = module.bias.data
                if delta.shape != base.shape:
                    raise ValueError(
                        f"LoRA bias shape {tuple(delta.shape)} does not match "
                        f"{tuple(base.shape)}")
                base += delta.to(base.device, base.dtype) * multiplier
                continue

            base = module.weight.data
            if kind == "diff":
                _, delta, multiplier = update
                if delta.shape != base.shape:
                    raise ValueError(
                        f"LoRA diff shape {tuple(delta.shape)} does not match "
                        f"{tuple(base.shape)}")
                base += delta.to(base.device, base.dtype) * multiplier
                continue

            _, up, down, scale = update
            delta = _compute_lora_delta(
                up.to(base.device, torch.float32),
                down.to(base.device, torch.float32),
                base)
            if delta.shape != base.shape:
                raise ValueError(
                    f"LoRA delta shape {tuple(delta.shape)} does not match "
                    f"{tuple(base.shape)}")
            base += delta.to(base.dtype) * scale
