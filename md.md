python gs2colmap/select_object.py \
    --ply /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/microwave/microwave.ply \
    --output /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/microwave/traj.json

python gs2colmap/render.py \
    --ply /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/microwave/microwave.ply \
    --trajectory /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/microwave/traj.json \
    --output /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/microwave \
    --width 640 \
    --height 480 \
    --fovy 65.0

python gs2colmap/sam3_video_debug.py gs2colmap/ply_data/microwave/images -o gs2colmap/ply_data/microwave/sam_results \
    --prompt "washing machine door" \
    --start-frame 13 

python visualize_axis.py \
  --ply /home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets/globe_with_stand/render_output/globe_with_stand_seg_background_clean.ply \
  --axis-json /home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets/globe_with_stand/render_output/globe_with_stand_seg_axis.json

python visualize_axis.py \
  --ply /home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets/laptop/render_output/laptop_seg_gs.ply \
  --axis-json /home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets/laptop/render_output/laptop_seg_axis.json



python sam3_video_debug.py images -o results \
    --prompt " door" 

python gs2colmap/segment_gaussian.py \
    --ply /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/washing/washing.ply \
    --masks /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/washing/sam_results/masks \
    --transforms /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/washing/transforms.json \
    --output /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/washing/washing_seg.ply \
    --mode vote --vote-threshold 0.8 
    # --mode intersection
    
python gs2colmap/render.py \
     --ply /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/washing/washing.ply \
     --trajectory /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/washing/traj \
     --output /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data/washing \
    --fovy 65.0

python gs2colmap/select_object.py \
    --ply gs2colmap/gs_ply/point_cloud.ply \
    --output gs2colmap/gs_ply/trajectory.json




python gs2colmap/render.py \
    --ply gs2colmap/gs_ply/point_cloud.ply \
    --trajectory gs2colmap/gs_ply/trajectory.json \
    --output gs2colmap/gs_ply/processed_data \
    --width 640 \
    --height 480 \
    --fovy 65.0


python gs2colmap/select_object.py \
    --ply gs2colmap/xiaomi_ply/point_cloud_seg_res.ply \
    --output gs2colmap/xiaomi_ply/trajectory.json \
    --num-views 100

python gs2colmap/tsdf_fusion.py \
    --render-dir gs2colmap/xiaomi_ply/processed_data \
    --output gs2colmap/xiaomi_ply/reconstruction.ply


python sdf/scripts/train.py bakedsdf --vis wandb \
    --output-dir outputs/xiaomi_ply --experiment-name xiaomi_sdf_recon \
    --trainer.steps-per-eval-image 2000 --trainer.steps-per-eval-all-images 250001 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 250001 \
    --optimizers.fields.scheduler.max-steps 250000 \
    --optimizers.field-background.scheduler.max-steps 250000 \
    --optimizers.proposal-networks.scheduler.max-steps 250000 \
    --pipeline.model.eikonal-anneal-max-num-iters 250000 \
    --pipeline.model.beta-anneal-max-num-iters 250000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm inf \
    --pipeline.model.mono-normal-loss-mult 0.5 \
    --pipeline.model.mono-depth-loss-mult 1.0 \
    --pipeline.model.near-plane 1e-6 \
    --pipeline.model.far-plane 10 \
    panoptic-data \
    --data /home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/xiaomi_ply/processed_data \
    --panoptic_data False \
    --mono_normal_data True \
    --mono_depth_data True \
    --panoptic_segment False \
    --downscale_factor 1 \
    --num_max_image 2000 