import os
import json
from lit_llama.tokenizer import Tokenizer
data_paths=['data/OGSG_data/initsg_planning_caption',
        'data/OGSG_data/initsg_planning_long_caption',    
'data/OGSG_data/initsg_finalsg_planning',
'data/OGSG_data/initsg_finalsg_targetobj_planning',
'data/OGSG_data/initsg_planning',
'data/OGSG_data/initsg_targetobj_planning']



tokenizer_path="checkpoints/lit-llama/tokenizer.model"
tokenizer = Tokenizer(tokenizer_path)
for data_path in data_paths:
    with open(os.path.join(data_path,f'{data_path.split("/")[-1]}_train.json'),"r")as f:
        data=json.load(f)
    token_len=0
    for key,info in data.items():
        token_len+=len(tokenizer.encode(info['instruction']))
    avg_token=token_len/len(data)

    print(f"Average token length for {data_path} is {avg_token}")