version=$(date +"%m_%d_%s")
EXP_NAME="initsg_planning_caption"
PROJECT_NAME="octopus_adapter_${version}_${EXP_NAME}"

export WANDB_API_KEY="36f2c08bcd41ad272c2e92b45b5c271985f01cf4"
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=SG_VLM_trail2

wandb online

# ["initsg_planning","initsg_targetobj_planning","initsg_finalsg_planning","initsg_finalsg_targetobj_planning"]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune/OGSG_adapter.py $EXP_NAME $PROJECT_NAME


