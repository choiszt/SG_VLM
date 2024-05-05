import os
import openai
from tqdm import tqdm
from bddl.activity import Conditions 

import chromadb
import re

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
            else:
                rel[1]=rel[1].lstrip("?")
            if not rel[2].endswith("_1"):
                rel[2]=(rel[2]+"_1").lstrip("?")        
            else:
                rel[2]=rel[2].lstrip("?")
            if (rel[1] not in objects) or (rel[2] not in objects):
                continue
            final.append([rel[1],rel[0],rel[2]])
    return final

chroma_client = chromadb.PersistentClient(path="EmbodiedMapping_database")

data_path="data/Omnigibson_sg_JSON/test"
task_base_path="Inference_result/initsg_planning"
OGSG_result={}
failed_list=[]
for act in tqdm(os.listdir(data_path)):
    bddl_task=act.split(".json")[0] 

    target_path=os.path.join(task_base_path,act.replace(".json",".txt"))
    task_raw_data=open(target_path).read()
    
    behavior_activity =bddl_task 
    activity_definition = 0                      
    simulator = "omnigibson"
    conds = Conditions(behavior_activity, activity_definition, simulator)

    objects=[a+"_1" for a in conds.parsed_objects.keys()] #Get the first instance 
    goal=get_scene_graph(conds.parsed_goal_conditions,objects)

    pattern = r"Step \d+: ([^\n]+)"
    steps = re.findall(pattern, task_raw_data)

    try:
        collection = chroma_client.get_collection(name=f"{bddl_task}")

    except:
        print(f"Don't have collection {bddl_task} yet")

    if steps:
        results = collection.query(
        query_texts=steps,
        n_results=1)
    else:
        failed_list.append(bddl_task)
        continue
    
    if results['ids']:
        header=f"def {bddl_task}():"
        with open(f"EmbodiedCodeMapping/code/{bddl_task}.py","w+")as f:
            f.write(header)
            f.write("\n")
        for code in results['ids']:
            with open(f"EmbodiedCodeMapping/code/{bddl_task}.py","a+")as f:
                f.write(f"\t{code[0]}")
                f.write("\n")
        with open(f"EmbodiedCodeMapping/code/{bddl_task}.py","a+")as f:
            f.write("\'''")
            f.write("\n")
            f.write(f"{str(goal)}")
            f.write("\n")
            f.write("\'''")