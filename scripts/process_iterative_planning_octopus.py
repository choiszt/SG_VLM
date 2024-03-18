
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

def find_last_continuous_ones_index(lst):
    start = None
    end = None
    if lst[-1]==0:
        return None
    for i in range(len(lst) - 1, -1, -1):  
        if lst[i] == 1: 
            if end is None: 
                end = i 
            start = i 
        else:  
            if end is not None: 
                break  
    return start, end

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
    reward_list={}

    for k in octopus_data['data'].keys():
        one_data=octopus_data['data'][k]
        task_status = '_'.join(k.split('_')[:-2])

        reward_list[task_status]={}

        if task_status not in status_list: #new task
            status_list.append(task_status)
            temp_reward=[]
        temp_reward.append(one_data['reward'])
        reward_list[task_status]['reward']=temp_reward
    
    succeed_task=[]
    final_reward_list={}
    for task in list(reward_list.keys()):
        if 'succeed'in task:
            if find_last_continuous_ones_index(reward_list[task]['reward'])==None: #the task fail
                continue
                print(task)
            else:
                start,end=find_last_continuous_ones_index(reward_list[task]['reward'])
                final_reward_list[task]=reward_list[task]
                final_reward_list[task]['index']=[start,end]

    tapa_octopus={}
    for k in final_reward_list.keys():
        tapa_octopus[k]={}
        plan_list=[]
        SG_list=[]
        temp_reward=[]
        
        start_subtask=final_reward_list[k]['index'][0]
        end_subtask=final_reward_list[k]['index'][1]
        for i in range(start_subtask,end_subtask+1):
            tail=f"_subtask_{i+1}"
            one_data=octopus_data['data'][k+tail]
            code=parse_code(one_data['answer'])
            executable_plan=code.split("\n")[1].lstrip().split(":")[1].lstrip()
            plan_list.append(executable_plan)
            SG_list.append(one_data['relations'].split(":")[1].lstrip()[:-1])   

            tapa_octopus[k]['plan']=plan_list
            tapa_octopus[k]['goal']=parse_goal(one_data['instruction'])[:-1].split(":")[1].lstrip()  # 'Task Goal: fold_a_towl_and_put_it_in_the_basket\n'
            tapa_octopus[k]['SceneGraph']=SG_list        # Scene Graph

    filtered_tapa_octopus={}
    falut={}
    for key in tapa_octopus.keys():
        if all(item=="None" for item in tapa_octopus[key]['SceneGraph']):
            # a=tapa_octopus[key]
            falut[key]=tapa_octopus[key] # 74 task
        else:
            filtered_tapa_octopus[key]=tapa_octopus[key]  # 302 task


    result={}
    for key in list(filtered_tapa_octopus.keys()):

        task=filtered_tapa_octopus[key]
        Previous_plan=''
        Previous_sg=''
        for cnt in tqdm(range(len(task["plan"]))):
            task_with_step=f"{key}_step{cnt+1}"
            result_plan=''
            result[task_with_step]={}
            goal=task['goal']
            plan=task['plan']

            prompt="Now, please output the next step of the plan to achieve the next subtask.\n"

            if cnt==0:
                Previous_plan=f"Previous Plan:\nNone\n"
                Previous_sg=f"Previous SceneGraph: None\n"
                Scene_Graph=f"Scene Graph: {task['SceneGraph'][cnt]}\n"
                result[task_with_step]['instruction']=f"Task Goal: {goal}\n{Previous_plan}{Previous_sg}{Scene_Graph}{prompt}"

                result_plan+=f"{task['plan'][0]}\n"
                # for i in range(len(task['plan'])):
                #     result_plan+=f"Step {i+1}: {task['plan'][i]}\n"
                result[task_with_step]['answer']=result_plan

            else:
                pre_plan=''
                for i in range(0,cnt):
                    pre_plan+=f"Step {i+1}: {task['plan'][i]}\n"  

                Previous_plan=f"Previous Plan:\n{pre_plan}\n"
                Previous_sg=f"Previous SceneGraph: {task['SceneGraph'][cnt-1]}\n"
                Scene_Graph=f"Scene Graph: {task['SceneGraph'][cnt]}\n"
                result[task_with_step]['instruction']=f"Task Goal: {goal}\n{Previous_plan}{Previous_sg}{Scene_Graph}{prompt}"
                
                result_plan+=f"{task['plan'][cnt]}\n"
                result[task_with_step]['answer']=result_plan

    with open("./data/Octopus/Octopus_iterative_executable_planning.json","w+")as f:
        f.write(json.dumps(result))



