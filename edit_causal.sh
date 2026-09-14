#!/usr/bin/env bash
# EditaLive streaming (causal) inference.
export PYTHONNOUSERSITE=1

CKPT_DIR=./weights/Wan-Animate
LORA_EDIT=./weights/EditaLive/editalive_edit.safetensors
LORA_LIGHTX2V=./weights/EditaLive/lightx2v.safetensors
LORA_DMD=./weights/EditaLive/editalive_streaming.safetensors

INPUT_JSON=./demo/edit_causal.json

python edit_streaming.py \
  --ckpt_dir "${CKPT_DIR}" \
  --lora_paths "${LORA_EDIT}" "${LORA_LIGHTX2V}" "${LORA_DMD}" \
  --input_json "${INPUT_JSON}" \
  --frame_num -1 \
  --sample_solver unipc \
  --fps 16 \
  --short_edge 480 \
  --long_edge 832 \
  --enable_compile False \
  --fast_decode True \
  --save_dir ./outputs_streaming
