### Polygon CGH with textured meshes
output_dir="results/"

### Mip-NeRF 360 ###
all_scenes=(garden kitchen room counter bicycle stump bonsai)
paper_scenes=(garden kitchen room counter bicycle stump bonsai)

for scene in "${paper_scenes[@]}"; do
  CUDA_VISIBLE_DEVICES=5 python3 main.py \
    --target_asset=textured_mesh \
    --method=silhouette \
    --num_frames=1 \
    --channel=3 \
    --gs_model_path="models/mipnerf360_default/${scene}/1000000/ckpts/ckpt_29999.pt" \
    --data_path="models/processed_textured_mesh_cgh_v2/${scene}.npz" \
    --out_path="${output_dir}/polygon_sil/${scene}" \
    --scene_dir="data/360_v2/${scene}" \
    --out_resolution_hologram 1024 1536 \
    --no_alpha_wave_blending \
    --resolution_scale_factor 2
done

### Blender ###
all_scenes=(hotdog lego ficus drums ship mic materials chair)
paper_scenes=(hotdog lego ficus drums ship mic materials chair)

for scene in "${paper_scenes[@]}"; do
  python3 main.py \
    --target_asset=textured_mesh \
    --method=silhouette \
    --num_frames=1 \
    --channel=3 \
    --gs_model_path="models/blender_default/${scene}/1000000/ckpts/ckpt_29999.pt" \
    --data_path="models/processed_textured_mesh_cgh_v2/${scene}.npz" \
    --out_path="${output_dir}/polygon_sil/${scene}" \
    --scene_dir="data/nerf_synthetic/${scene}" \
    --out_resolution_hologram 1024 1024 \
    --no_alpha_wave_blending \
    --resolution_scale_factor 2
done