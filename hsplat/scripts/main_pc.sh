### Point cloud CGH with points sampled from textured meshes
pc_scale_multiplier=1.0  # default = 1.0 (0.3, 0.5, 1.0, 2.0)
output_dir="results/"

### Mip-NeRF 360 ###
all_scenes=(garden kitchen room counter bicycle stump bonsai)
paper_scenes=(garden kitchen room counter bicycle stump bonsai)

for scene in "${paper_scenes[@]}"; do
    python3 main.py \
        --target_asset=points_from_mesh \
        --method=alpha_wave_blending \
        --num_frames=1 \
        --channel=3 \
        --gs_model_path="models/mipnerf360_default/${scene}/1000000/ckpts/ckpt_29999.pt" \
        --data_path="models/processed_textured_mesh_cgh_v2/${scene}.npz" \
        --out_path="${output_dir}/pc_from_mesh_${pc_scale_multiplier}/${scene}" \
        --scene_dir="data/360_v2/${scene}" \
        --out_resolution_hologram 1024 1536 \
        --resolution_scale_factor 2 \
        --pc_scale_multiplier $pc_scale_multiplier
done

### Blender ###
all_scenes=(hotdog lego ficus drums ship mic materials chair)
paper_scenes=(hotdog lego ficus drums ship mic materials chair)

for scene in "${paper_scenes[@]}"; do
    python3 main.py \
        --target_asset=points_from_mesh \
        --method=alpha_wave_blending \
        --num_frames=1 \
        --channel=3 \
        --gs_model_path="models/blender_default/${scene}/1000000/ckpts/ckpt_29999.pt" \
        --data_path="models/processed_textured_mesh_cgh_v2/${scene}.npz" \
        --out_path="${output_dir}/pc_from_mesh_${pc_scale_multiplier}/${scene}" \
        --scene_dir="data/nerf_synthetic/${scene}" \
        --out_resolution_hologram 1024 1024 \
        --resolution_scale_factor 2 \
        --pc_scale_multiplier $pc_scale_multiplier
done