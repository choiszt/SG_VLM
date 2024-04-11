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

# if __name__ == '__main__':

train_path="data/Omnigibson_sg_JSON/train"
test_path="data/Omnigibson_sg_JSON/test"
errorlist=[]

for task in tqdm(os.listdir(train_path)):
    with open(os.path.join(train_path,task),"r")as f:
        og_planning=json.load(f)

    bddl_task=task.split(".json")[0] 

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

    input_initial_scene_graph="\n".join(str(tuple(a)) for a in initial_scene_graph)
    input_goal_scene_graph="\n".join(str(tuple(a)) for a in goal_scene_graph)

#Generate Planning
planning=''
for step,info in og_planning.items():
    planning+=f"{step}: {info['Planning']}\n"


    status_list=[]
    plan_list=[] #store the executable plan
    task_plan={}
    for k in octopus_data['data'].keys():
        one_data=octopus_data['data'][k]
        task_status = '_'.join(k.split('_')[:-3])

        tapa_octopus[task_status]={}

        if task_status not in status_list: #new task
            status_list.append(task_status)
            plan_list=[]
            SG_list=[]

        if one_data['reward']==1: #denote that the code is success -> the plan is valid
            code=parse_code(one_data['answer'])
            try:
                executable_plan=code.split("\n")[1].lstrip().split(":")[1].lstrip()
                plan_list.append(executable_plan)
                SG_list.append(one_data['relations'].split(":")[1].lstrip()[:-1])
                # task_plan[task_status]=plan_list

                tapa_octopus[task_status]['plan']=plan_list
                tapa_octopus[task_status]['goal']=parse_goal(one_data['instruction'])[:-1].split(":")[1].lstrip()  # 'Task Goal: fold_a_towl_and_put_it_in_the_basket\n'
                tapa_octopus[task_status]['SceneGraph']=SG_list        # Scene Graph
                # tapa_octopus[task_status]['gpt_planning']=parse_answer(one_data['answer']) #Explain + New Planning
            except Exception as e:
                continue

    print("checkmark") #60 empty and 421 valid executable planning task

    result={}
    for key in list(tapa_octopus.keys()):
        if tapa_octopus[key]=={}:
            continue
        task=tapa_octopus[key]
        Previous_plan=''
        Previous_sg=''
        for cnt in tqdm(range(len(task["plan"]))):
            task_with_step=f"{key}_step{cnt+1}"
            result_plan=''
            result[task_with_step]={}
            goal=task['goal']
            plan=task['plan']
            if cnt==0:
                Previous_plan=f"Previous Plan: None\n"
                Previous_sg=f"Previous SceneGraph: None\n"
                Scene_Graph=f"Scene Graph: {task['SceneGraph'][cnt]}\n"
                result[task_with_step]['instruction']=f"Task Goal: {goal}\n{Previous_plan}{Previous_sg}{Scene_Graph}"
                for i in range(len(task['plan'])):
                    result_plan+=f"Step {i+1}: {task['plan'][i]}\n"
                result[task_with_step]['answer']=result_plan

            else:
                Previous_plan=f"Previous Plan: {task['plan'][cnt-1]}\n"
                Previous_sg=f"Previous SceneGraph: {task['SceneGraph'][cnt-1]}\n"
                Scene_Graph=f"Scene Graph: {task['SceneGraph'][cnt]}\n"
                result[task_with_step]['instruction']=f"Task Goal: {goal}\n{Previous_plan}{Previous_sg}{Scene_Graph}"
                for i in range(cnt,len(task['plan'])):
                    result_plan+=f"Step {i+1}: {task['plan'][i]}\n"
                result[task_with_step]['answer']=result_plan

    with open("./data/Octopus/Octopus_executable_planning.json","w+")as f:
        f.write(json.dumps(result))



