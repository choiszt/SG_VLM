#!/bin/bash
cd /mnt/bn/vl-research/workspace/yhzhang/LLaVA_dev

# Get the installed version of transformers
installed_version=$(pip show transformers | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "4.36.2" ]; then
    pip install transformers==4.36.2
fi

# Get the installed version of deepspeed
installed_version=$(pip show deepspeed | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.12.6" ]; then
    pip install deepspeed==0.12.6
fi

# Install ninja if not installed
if ! pip show ninja > /dev/null 2>&1; then
    pip install ninja
fi

# Install flash-atten if not installed
if ! pip show flash-attn > /dev/null 2>&1; then
    pip install flash-attn --no-build-isolation
fi

# Install decord if not installed
if ! pip show decord > /dev/null 2>&1; then
    pip install decord
fi

# Install protobuf if not installed
if ! pip show protobuf > /dev/null 2>&1; then
    pip install protobuf 
fi

pip install -U wandb


################## project ##################
PROJECT_NAME="llava-vicuna_7B-mlp2x_gelu-llava_558k_with_webvid-spatial_pool4-video_fps1-pt"


# wandb configure
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=LLaVA_v1.6_video

wandb offline

deepspeed --master_port 29501 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version plain_guided \
    --data_path ./data/LLaMA-VID-Pretrain/llava_558k_with_webvid_subset.json \
    --image_folder ./data/LLaMA-VID-Pretrain/images \
    --video_folder ./data/LLaMA-VID-Pretrain/videos \
    --vision_tower  openai/clip-vit-large-patch14 \
    --image_processor openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./work_dirs/$PROJECT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --mm_resampler_type qformer \
    --mm_qformer_depth 2 \
    --mm_qformer_latents 32 \
    --mm_qformer_pretrained model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth \


    #./model_zoo/LAVIS/eva_vit_g.pth \
    # --image_aspect_ratio anyres \
    # --image_grid_pinpoints "[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]" \
    # --mm_patch_merge_type spatial_unpad \\