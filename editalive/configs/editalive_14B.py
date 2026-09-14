import torch
from easydict import EasyDict

editalive_14B = EasyDict(__name__='Config: EditaLive 14B')

# ---------------------------------------------------------------- weights --- #
# Sub-paths inside --ckpt_dir.
editalive_14B.transformer_subpath = './'
editalive_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
editalive_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
editalive_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
editalive_14B.t5_tokenizer = 'google/umt5-xxl'

# UMT5-XXL encoder geometry. Needed because the T5 checkpoint is a bare state
# dict with no config.json alongside it, unlike the transformer.
editalive_14B.t5_kwargs = EasyDict(
    vocab=256384,
    dim=4096,
    dim_attn=4096,
    dim_ffn=10240,
    num_heads=64,
    num_layers=24,
    num_buckets=32,
    shared_pos=False,
    dropout=0.0,
)

# --------------------------------------------------------------- sampling --- #
# dtype the transformer and text encoder are loaded in
editalive_14B.param_dtype = torch.bfloat16

# flow-matching schedule
editalive_14B.num_train_timesteps = 1000
editalive_14B.sample_shift = 5.0
editalive_14B.sample_steps = 2
editalive_14B.sample_guide_scale = 1.0

# clip defaults
editalive_14B.frame_num = -1
editalive_14B.prompt = '视频中的人在做动作'

# -------------------------------------------------------------- streaming --- #
# The rolling KV cache keeps `local_attn_size` latent frames of context, and the
# model consumes `chunk_size` latent frames per step.
editalive_14B.local_attn_size = 6
editalive_14B.chunk_size = 3
