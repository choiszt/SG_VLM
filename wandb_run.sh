version=$(date +"%m_%d_%s")
EXP_NAME="initsg_planning"
PROJECT_NAME="octopus_adapter_${version}_${EXP_NAME}"

export WANDB_API_KEY="36f2c08bcd41ad272c2e92b45b5c271985f01cf4"
wandb login $WANDB_API_KEY

export WANDB_NAME=$PROJECT_NAME

export WANDB_PROJECT=SG_VLM

wandb online
# ["initsg_planning","initsg_targetobj_planning","initsg_finalsg_planning","initsg_finalsg_targetobj_planning"]
srun -p 3dobject_aigc_mid --gres=gpu:8 -J 3d_Caption --ntasks-per-node=8 python finetune/OGSG_adapter.py $EXP_NAME
