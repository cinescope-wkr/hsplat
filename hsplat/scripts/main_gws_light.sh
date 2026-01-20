#!/usr/bin/env bash
set -euo pipefail

num_gs=10000
output_dir="results"

paper_scenes=(lego)

for scene in "${paper_scenes[@]}"; do
  python3 main.py \
    --target_asset=gaussians \
    --method=alpha_wave_blending \
    --num_frames=1 \
    --channel=3 \
    --gs_model_path="models/blender_default/${scene}/1000000/ckpts/ckpt_29999.pt" \
    --data_path="models/blender_default/${scene}/${num_gs}/ckpts/ckpt_29999.pt" \
    --out_path="${output_dir}/full_awb_${num_gs}/${scene}" \
    --scene_dir="data/NeRF_Data/nerf_synthetic/${scene}" \
    --out_resolution_hologram 1024 1024 \
    --resolution_scale_factor 2
done