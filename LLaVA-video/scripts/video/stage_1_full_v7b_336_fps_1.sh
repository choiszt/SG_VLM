#!/bin/bash
DIR="/mnt/bn/vl-research/workspace/yhzhang/llava-video-old"

# Check if the directory exists
if [ -d "$DIR" ]; then
    # If the directory exists, set BYTENAS to "vl-research"
    BYTENAS="vl-research"
else
    # If the directory does not exist, set BYTENAS to "vl-research-cn-lq"
    BYTENAS="vl-research-cn-lq"

    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export HF_HOME=/mnt/bn/vl-research-cn-lq/.cache/huggingface
fi

DIR=/mnt/bn/${BYTENAS}/workspace/yhzhang/llava-video-old

cd ${DIR}

pwd

pip3 install --upgrade pip

# Get the installed version of transformers
installed_version=$(pip3 show transformers | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "4.38.2" ]; then
    pip3 install transformers==4.38.2
fi

# Get the installed version of deepspeed
installed_version=$(pip3 show deepspeed | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.12.2" ]; then
    pip3 install deepspeed==0.12.2
fi

# Install ninja if not installed
if ! pip3 show ninja > /dev/null 2>&1; then
    pip3 install ninja
fi

# Install flash-atten if not installed
if ! pip3 show flash-attn > /dev/null 2>&1; then
    pip3 install flash-attn --no-build-isolation
fi

# Install decord if not installed
if ! pip3 show decord > /dev/null 2>&1; then
    pip3 install decord
fi

# Install protobuf if not installed
if ! pip3 show protobuf > /dev/null 2>&1; then
    pip3 install protobuf 
fi

# Install torchvision if not installed
if ! pip3 show torchvision > /dev/null 2>&1; then
    pip3 install torchvision==0.16.0
fi

# Install timm if not installed
if ! pip3 show timm > /dev/null 2>&1; then
    pip3 install timm
fi

# Get the installed version of transformers
installed_version=$(pip3 show accelerate | grep Version | cut -d ' ' -f 2)

# Check if the installed version is not the latest
if [ "$installed_version" != "0.27.2" ]; then
    pip3 install accelerate==0.27.2
fi


# Install sentencepiece if not installed
if ! pip3 show sentencepiece > /dev/null 2>&1; then
    pip3 install sentencepiece==0.1.99
fi



POOL_STRIDE=$1

if [ "$POOL_STRIDE" -lt 4 ]; then
    LLM="vicuna-7b-v1-5-8k"
else
    LLM="vicuna-7b-v1-5"
fi


echo ${LLM}
################## project ##################
PROJECT_NAME=llava-vicuna_7B-mlp2x_gelu-llava_558k_with_webvid-spatial_pool${POOL_STRIDE}-video_fps1-336px-pt


# wandb configure
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=LLaVA_v1.6_video

wandb offline

deepspeed --master_port 29501 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/bn/${BYTENAS}/checkpoints/vicuna/${LLM} \
    --version plain_guided \
    --data_path /mnt/bn/${BYTENAS}/workspace/yhzhang/llava-video/data/llava_video/meta/pretrain/llava_558k_with_webvid.json \
    --image_folder /mnt/bn/${BYTENAS}/data/llava_data/blip_558k/images/ \
    --video_folder /mnt/bn/${BYTENAS}/data/llava_video/WebVid/videos_subset \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --image_processor openai/clip-vit-large-patch14-336 \
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
    --dataloader_num_workers 4 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_stride ${POOL_STRIDE} \
    --mm_spatial_pool_out_channels 1024 \

    #./model_zoo/LAVIS/eva_vit_g.pth \
    # --image_aspect_ratio anyres \
    # --image_grid_pinpoints "[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]" \
    # --mm_patch_merge_type spatial_unpad \
