export CUDA_VISIBLE_DEVICES=0
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

torchrun --standalone --nnodes 1 --nproc-per-node 1 --master_port 29508 vla-scripts/train.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/ \
  --dataset_name fractal20220817_data \
  --run_root_dir /data/wandbrun \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --use_proprio False \
  --num_images_in_input 1 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --shuffle_buffer_size 56_000 \
  --num_steps_before_decay 100000 \
  --max_steps 200005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_project yourproject \
  --wandb_entity yourname \
  --num_actions_chunk 8 \
  --num_actions_per_token 8 \
  --num_blocks 2 \
  --mode "mul" \
  --action_head_name "funnel" 


# for bridge
#   --dataset_name bridge_orig \