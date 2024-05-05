import os
import json
EXPs=[ 
'initsg_finalsg_planning',
'initsg_finalsg_targetobj_planning',
'initsg_planning',
'initsg_targetobj_planning']
step_cnt=0
for exp in EXPs:
    step_cnt=0
    for file in os.listdir(f"Inference_result/{exp}"):
        with open(f"Inference_result/{exp}/{file}","r")as f:
            data=f.read()
        step_cnt+=len(data.split("Step"))-1
    print(f"Average step count for {exp} is {step_cnt/len(os.listdir(f'Inference_result/{exp}'))}")
        