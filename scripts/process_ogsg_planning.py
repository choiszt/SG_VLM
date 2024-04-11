import ptvsd
ptvsd.enable_attach(address=('10.140.0.184', 5678))
ptvsd.wait_for_attach()
import re
import json
from tqdm import tqdm
import os
import json
import numpy as np
from bddl.activity import Conditions 

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

def convert_txt_to_JSON():
    data_path="data/Omnigibson_sg_planning"
    target_path="data/Omnigibson_sg_JSON"
    fault_list=[]
    for activity in tqdm(os.listdir(data_path)):
        tar_path=os.path.join(data_path,activity)
        json_path=f"{target_path}/{activity.split('.')[0]}.json"
        try:
            with open(tar_path,"r")as f:
                data=json.loads(f.read())
            with open(json_path,"w")as f:
                f.write(json.dumps(json.loads(data)))
        except:
            fault_list.append(activity)
            continue
    return fault_list

def log_failed_bddl(fault_list):
    for fail_task in fault_list:
        with open("data/Omnigibson_sg_JSON/fault_list/fail_task.txt","a+")as f:
            f.write(fail_task)
            f.write("\n")

def split_dataset(ratio=0.85):
    import math,random,shutil
    list_dataset=[a for a in os.listdir("data/Omnigibson_sg_JSON")if a.endswith(".json")]

    random.shuffle(list_dataset)

    train_path="data/Omnigibson_sg_JSON/train"
    test_path="data/Omnigibson_sg_JSON/test"
    makedir=lambda PATH: os.makedirs(PATH) if not os.path.exists(PATH) else None
    makedir(train_path)
    makedir(test_path)

    base_path="data/Omnigibson_sg_JSON"

    dataset_size=len(list_dataset)
    train_size=math.floor(ratio*dataset_size)
    #random rample:
    test_index=[]
    train_index=sorted(np.random.choice(dataset_size, size=train_size, replace=False))
    for i in range(dataset_size):
        if i not in train_index:
            test_index.append(i)

    try:
        #move train dataset
        for idx in train_index:
            file_path=list_dataset[idx]
            shutil.move(os.path.join(base_path,file_path),os.path.join(train_path,file_path))
        #move test dataset
        for idx in test_index:
            file_path=list_dataset[idx]
            shutil.move(os.path.join(base_path,file_path),os.path.join(test_path,file_path))
    except:
        AssertionError

def remove_null_task(train_path,test_path):
    for task in os.listdir(train_path):
        try:
            with open(os.path.join(train_path,task),"r")as f:
                og_planning=json.load(f)
        except:
            os.remove(os.path.join(train_path,task)) #remove null file

    for task in os.listdir(test_path):
        try:
            with open(os.path.join(test_path,task),"r")as f:
                og_planning=json.load(f)
        except:
            os.remove(os.path.join(test_path,task)) #remove null file    


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

def generate_instruction(option,is_train):
    if is_train:
        dataset_path="data/Omnigibson_sg_JSON/train"
    else:
        dataset_path="data/Omnigibson_sg_JSON/test"
    errorlist=[]

    OGSG_result={}
    for task in tqdm(os.listdir(dataset_path)):
        with open(os.path.join(dataset_path,task),"r")as f:
            og_planning=json.load(f)


        bddl_task=task.split(".json")[0] 
        OGSG_result[bddl_task]={}

        behavior_activity =bddl_task 

        activity_definition = 0                      
        simulator = "omnigibson"
        conds = Conditions(behavior_activity, activity_definition, simulator)

        objects=[a+"_1" for a in conds.parsed_objects.keys()] #Get the first instance 
        initial=conds.parsed_initial_conditions
        initial_scene_graph=get_scene_graph(initial,objects)

        clean_objects=[obj.split(".")[0] for obj in objects]

        goal=conds.parsed_goal_conditions
        goal_scene_graph=get_scene_graph(goal,objects)

        input_initial_scene_graph="\n".join(str(list(a)) for a in initial_scene_graph)
        input_goal_scene_graph="\n".join(str(list(a)) for a in goal_scene_graph)

        if option=="initsg_planning":
            #Generate Planning
            planning=''
            for step,info in og_planning.items():
                planning+=f"{step}: {info['Planning']}\n"     

            OGSG_result[bddl_task]['instruction']=f"Task Goal:\n{bddl_task}\nObserved Relation:\n{input_initial_scene_graph}\nNow please output plannings for doing {bddl_task}"
            OGSG_result[bddl_task]['answer']=planning

        if option=="initsg_targetobj_planning":
            #Generate Planning
            planning=''
            for step,info in og_planning.items():
                planning+=f"{step}: {info['Planning']}\nTarget:{str(info['Target'])}\n"                 

            OGSG_result[bddl_task]['instruction']=f"Task Goal:\n{bddl_task}\nObserved Relation:\n{input_initial_scene_graph}\nNow please output plannings for doing {bddl_task}"
            OGSG_result[bddl_task]['answer']=planning

        if option=="initsg_finalsg_planning":
            planning=''
            for step,info in og_planning.items():
                planning+=f"{step}: {info['Planning']}\n"    

            OGSG_result[bddl_task]['instruction']=f"Task Goal:\n{bddl_task}\nObserved Relation:\n{input_initial_scene_graph}\Goal Expected Relation:\n{input_goal_scene_graph}\nNow please output plannings for doing {bddl_task}"
            OGSG_result[bddl_task]['answer']=planning

        if option=="initsg_finalsg_targetobj_planning":
            planning=''
            for step,info in og_planning.items():
                planning+=f"{step}: {info['Planning']}\nTarget:{str(info['Target'])}\n"     

            OGSG_result[bddl_task]['instruction']=f"Task Goal:\n{bddl_task}\nObserved Relation:\n{input_initial_scene_graph}\Goal Expected Relation:\n{input_goal_scene_graph}\nNow please output plannings for doing {bddl_task}"
            OGSG_result[bddl_task]['answer']=planning

    return OGSG_result         

if __name__ == '__main__':
    makedir=lambda PATH: os.makedirs(PATH) if not os.path.exists(PATH) else None

    OPTIONS=["initsg_planning","initsg_targetobj_planning","initsg_finalsg_planning","initsg_finalsg_targetobj_planning"]
    for OPTION in OPTIONS:
        tar_path=os.path.join("data/OGSG_data/",OPTION)
        makedir(tar_path)
        OGSG_result=generate_instruction(OPTION,is_train=True)
        with open(f"{tar_path}/{OPTION}_train.json","w+")as f:
            f.write(json.dumps(OGSG_result))

        OGSG_result=generate_instruction(OPTION,is_train=False)
        with open(f"{tar_path}/{OPTION}_test.json","w+")as f:
            f.write(json.dumps(OGSG_result))

