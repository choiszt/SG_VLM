import os 
import json
from bddl.activity import Conditions 
# cnt=0

def get_room(goal,behavior_activity): #Recursively find scene graph in bddl file, Max Depth==3
    result={}
    result[behavior_activity]=[]
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
        if rel[0]=="inroom":
            result[behavior_activity].append(rel[2])
    result[behavior_activity]=list(set(result[behavior_activity]))
    return result

final={}

for ele in os.listdir("data/Omnigibson_sg_JSON/train"):
    # jsfile=os.path.join("data/Omnigibson_sg_JSON/train",ele)
    # with open(jsfile,"r")as f:
    #     data=json.load(f)
    
    # cnt+=len(data)
    bddl_task=ele.split(".json")[0]

    behavior_activity =bddl_task 

    activity_definition = 0                      
    simulator = "omnigibson"
    conds = Conditions(behavior_activity, activity_definition, simulator)
    init=conds.parsed_initial_conditions
    get_room(init,behavior_activity)
    final.update(get_room(init,behavior_activity))

a=set()
for name,room in final.items():
    for i in room:
        a.add(i)
print(final)