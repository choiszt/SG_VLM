version=(date +"%m_%d_%s")
PROJECT_NAME="octopus_adapter_$version"

export WANDB_API_KEY="36f2c08bcd41ad272c2e92b45b5c271985f01cf4"
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=octopus_adapter

wandb online

CUDA_VISIBLE_DEVICES=4,5,6,7 python finetune/octopus_adapter.py