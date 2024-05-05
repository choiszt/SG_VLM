import os

import yaml

import bddl
import json
import openai
import time
from tqdm import tqdm

from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from create_dataset.prompt.prompt_utils_og_choiszt import *

openai.api_type = "azure"
openai.api_base = "https://voyager.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "Your Azure Api Key"
os.environ["OPENAI_API_KEY"] = "Your Azure Api Key"

from bddl.activity import Conditions 

import os
from tqdm import tqdm

RELATION=["overlaid",
"contains",
"under",
"inside",
"filled",
"saturated",
"nextto",
"attached",
"insource",
"draped",
"touching",
"covered",
"ontop",
]

def get_all_possible_relation():
    relations=[]
    for act in tqdm(os.listdir("bddl/bddl/activity_definitions")):
        if act.endswith("bddl"):
            continue
        behavior_activity =act  # the activity you want to try, full list in bddl/bddl/activity_definitions
        activity_definition = 0                         # the specific definition you want to use. As of BEHAVIOR100 2021, this should always be 0.
        simulator = "omnigibson"                        # this does not require an actual simulator, just a domain file (e.g. activity_definitions/domain_omnigibson.bddl). You can make your own if desired.

        conds = Conditions(behavior_activity, activity_definition, simulator)

        for ele in conds.parsed_initial_conditions:
            if len(ele)==3:
                relations.append(ele[0])
        for ele in conds.parsed_goal_conditions:
            if len(ele)==3:
                relations.append(ele[0])
    return list(set(relations))

def write_relation(possible_relation):
    candidate_relation=[]
    with open("create_dataset/relation.txt","w")as f:
        for rel in possible_relation:
            if rel!="and" and rel!="exists" and rel!="forall" and rel!="or" and rel!="exists" and rel!="inroom":
                candidate_relation.append(rel)
                f.write(f"\"{rel}\"")
                f.write(",")
                f.write("\n")

# possible_relation=get_all_possible_relation()
# candidate_relation=[]
# for rel in possible_relation:
#     if rel!="and" and rel!="exists" and rel!="forall" and rel!="or" and rel!="exists" and rel!="inroom":
#         candidate_relation.append(rel)

def gpt_request(messages):
    response = openai.ChatCompletion.create(
        engine="voyager",
        messages = messages,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )

    return response['choices'][0]['message']['content']

def get_scene_graph(goal,objects): #Recursively find scene graph in bddl file, Max Depth==3
    all_relations=[]
    for a in goal:
        if isinstance(a,list):
            if(len(a)==3):
                all_relations.append(a)
            for b in a:
                if isinstance(b,list):
                    if(len(b)==3):
                        all_relations.append(b)
                for c in b:
                    if isinstance(b,list):
                        if(len(c)==3):
                            all_relations.append(c)
                    for d in c:
                        if isinstance(d,list):
                            if(len(d)==3):
                                all_relations.append(d)
    final=[]
    for rel in all_relations:
        if rel[0] in RELATION:
            if not rel[1].endswith("_1"):
                rel[1]=(rel[1]+"_1").lstrip("?")   

            if not rel[2].endswith("_1"):
                rel[2]=(rel[2]+"_1").lstrip("?")        

            if (rel[1] not in objects) or (rel[2] not in objects):
                continue
            final.append([rel[1].split(".")[0],rel[0],rel[2].split(".")[0]])
    return final



def main(random_selection=False, headless=False, short_exec=False):
    
    for act in tqdm(os.listdir("data/Omnigibson_sg_JSON/test")):
        
        behavior_activity=act.split(".json")[0]
        # input_context ="Objects:\n"+str(tuple(clean_objects))+"\n"+"Task:\n"+behavior_activity+"\n"+"Initial Scene Graph:\n" + str(input_initial_scene_graph) + "\n"+"Goal Scene Graph\n" + str(input_goal_scene_graph)\
        # +"\nNow,please output your plannings"

        # # messages = get_prompt()
        # # messages=get_prompt_instruction()
        messages = [{"role": "system", "content": "You are an indoor service robot to help me with everyday tasks by giving action planning with at most 7 steps. The simpler, the better"}]
        # messages=get_prompt_instruction_v2()
        # # messages=[{"role": "system", "content":get_fewshot_sample()[0]['context']}]
        # # Add fewshot_samples
        # # samples = get_fewshot_sample()
        # # for sample_one in samples:
        # #     messages.append({"role": "user", "content": sample_one["context"]})
        # #     messages.append({"role": "assistant", "content": sample_one["response"]})
        input_context ="Task:\n"+behavior_activity+"\nNow,please output your plannings"
        messages.append({"role": "user", "content": input_context})

        max_tries=0
        while True:
            if max_tries>2:
                break
            try:
                data=gpt_request(messages)
                break
            except:
                max_tries+=1  
                time.sleep(4)
        with open(f"Inference_result/GPT/Goal/{behavior_activity}.txt","w")as f:
            f.write(data)




if __name__ == "__main__":
    main()
