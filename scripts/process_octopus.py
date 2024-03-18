
import re
import json
from tqdm import tqdm

def parse_goal(instructions):
    task_goal_match = re.search(r'Task Goal: (.+?)\n', instructions)
    if task_goal_match:
        task_goal = task_goal_match.group(0)
    return task_goal

def parse_answer(instructions):
    answer_planning=instructions.split('Code')[0].split("Subtask:\n")[1]
    # return "Subtask:\n"+answer_planning
    return answer_planning

def parse_planning(instructions):
    planning=instructions.split("Original Subtasks:")[1].split('Previous')[0].lstrip(" ").lstrip("\n")
    
    # return "Original Subtasks: \n"+planning
    return planning

def parse_relation(text):
    pattern = r"\('([^']+)', '([^']+)', '([^']+)'\)"

    matches = re.findall(pattern, text)

    return matches

def parse_code(text):
    code_pattern = r"Code:\n(.*?)(?:\n\n|$)"
    match = re.search(code_pattern, text, re.DOTALL)

    if match:
        code_block = match.group(1)
    else:
        code_block = "No code block found."
    return code_block

if __name__ == "__main__":

    with open("./data/Octopus/OctoGibson.json","r")as f:
        octopus_data=json.load(f)

    tapa_octopus={}

    # for k in octopus_data['data'].keys():
    #     tapa_octopus[k]={}
    #     one_data=octopus_data['data'][k]
    #     tapa_octopus[k]['goal']=parse_goal(one_data['instruction'])  # 'Task Goal: fold_a_towl_and_put_it_in_the_basket\n'
    #     tapa_octopus[k]['SceneGraph']=one_data['relations']          # Scene Graph
    #     # tapa_octopus[k]['current_planning']=parse_planning(one_data['instruction']) #Original Subtask
    #     # tapa_octopus[k]['instruction']=one_data['instruction']     #Don't need
    #     tapa_octopus[k]['answer_planning']=parse_answer(one_data['answer']) #Explain + New Planning
    #     tapa_octopus[k]['reward']=one_data['reward']
    #     tapa_octopus[k]['main_reward']=one_data['main_reward']     
    #     # tapa_octopus[k]['objects']=one_data['objects']             #Don't need 
        
    # with open("./data/Octopus/SG_planning_Octopus.json","w+")as f:
    #     f.write(json.dumps(tapa_octopus))

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



