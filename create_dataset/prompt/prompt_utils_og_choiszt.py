import os
import numpy as np
import json


def get_prompt(example=None):
    messages = [{"role": "system", "content": f"""You are an indoor service robot to help me with everyday tasks by giving low-level action list. 
                 You are provided with "Objects", "Task", "Initial Scene Graph" and "Goal Scene Graph"

All possible relations for scene graph are listed:
overlaid
contains
under
inside
filled
saturated
nextto
attached
insource
draped
touching
covered
ontop
                 
You should only utilizing the following atomic motions: 
                 
MoveBot(env, robot, object, camera):
Go to the subgoal object. This action is finished once the object is visible and reachable.
Augments:
- env: a const argument, don't need to change.
- robot: a const argument, don't need to change.             
- object: a string, the object to go to.
- camera: a const argument, don't need to change.     
                             
EasyGrasp(robot, object):
Pick up an object.
Augments:
- robot: a const argument, don't need to change.                   
- object: a string, the object to pick.

put_ontop(robot, object1, object2)
Put the object1 within the robot's hands onto object2
Augments:
- robot: a const argument, don't need to change.  
- object1: a string, the object to be put on.
- object2: a string,,the receptacle to put the object1 on.                 

put_inside(robot, object1, object2): 
Put the obj1 within the robot's hands inside obj2
Augments:
- robot: a const argument, don't need to change.  
- object1: a string, the object to be put inside.
- object2: a string,,the receptacle to put the object1 inside.

cook(robot,object): 
cook the given object.     
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be cooked.                             

burn(robot,object): 
burn the given object.     
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be burnt.  
                 
freeze(robot,object): 
freeze the given object.     
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be freezed.    

heat(robot,object): 
heat the given object.     
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be heated.                     
                                               
open(robot,object): 
Open an openable object.   
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be opened.  

close(robot,object): 
close an openable object.   
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be closed.                         
                 
fold(robot,object): 
fold an foldable object.   
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be folded.     

unfold(robot,object): 
unfold an foldable object.   
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be unfolded.  

toggle_on(robot,object): 
Toggle a toggleable object. 
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be toggled on.                                 
                 
toggle_off(robot,object): 
Toggle off a toggleable object. 
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be toggled off.                        

sliceObject(robot,object): 
Slice a sliceable object.
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be sliced.   

clean(robot,object)
Wash an object in the sink.
Augments:
- robot: a const argument, don't need to change.  
- object: a string, the object to be cleaned.   


(1) Generate operation instructions using provided objects with the actions that must be performed to complete the operating instructions;
(2) Generate expected scene graph for each step.               
(3) Do not generate any actions that cannot be executed with confidence;
                             
here is an example:
{get_fewshot_sample()[0]['context']}                                     
"""}]
    return messages

# (4) You should response with following format: 
# 1. Action:Atomic Functions 
#    Scene Graph: expected scene graph after doing the action (also list if don't change)
# 2. Action:Atomic Functions 
#    Scene Graph: expected scene graph after doing the action (also list if don't change)
# 3. Action:Atomic Functions 
#    Scene Graph: expected scene graph after doing the action (also list if don't change)     
def get_prompt_instruction(example=None):
    messages = [{"role": "system", "content": f"""You are an indoor service robot to help me with everyday tasks by giving action planning and expected scene graph. 
                 You are provided with "Objects", "Task", "Initial Scene Graph" and "Goal Scene Graph"

All possible relations for scene graph are listed:
overlaid
contains
under
inside
filled
saturated
nextto
attached
insource
draped
touching
covered
ontop
                 
All possible actions for planning:
                 
Move
Go to the subgoal object. This action is finished once the object is visible and reachable.
                             
EasyGrasp
Pick up an object.

put_ontop
Put the object1 within the robot's hands onto object2
            
put_inside
Put the obj1 within the robot's hands inside obj2
                 
cook
cook the given object.     
           
burn
burn the given object.     
                 
freeze
freeze the given object.     

heat
heat the given object.                
                                               
open
Open an openable object.   

close
close an openable object.                
                 
fold
fold an foldable object.   

unfold
unfold an foldable object.   

toggle_on
Toggle a toggleable object.                
                 
toggle_off
Toggle off a toggleable object.        

sliceObject
Slice a sliceable object.

clean
Wash an object in the sink.

(1) Generate operation planning using provided objects with the actions that must be performed to complete the operating instructions;
(2) Generate expected scene graph for each step.               
(3) Do not generate any actions and scene graph that cannot be executed with confidence;

here is an example in JSON format:
{get_fewshot_sample()[0]['context']}                                              
"""}]
    return messages

def get_prompt_instruction_v2(example=None):
    messages = [{"role": "system", "content": f"""You are an indoor service robot to help me with everyday tasks by giving action planning and expected scene graph with no more than 7 steps.
                 You are provided with "Objects", "Task", "Initial Scene Graph" and "Goal Scene Graph"

All possible relations for scene graph are listed:
overlaid
contains
under
inside
filled
saturated
nextto
attached
insource
draped
touching
covered
ontop
                 
(1) Generate operation planning using provided objects with the actions that must be performed to complete the operating instructions;             
(2) Do not generate any actions and scene graph that cannot be executed with confidence;
(3) Your solution should no more than 7 steps!                 
(4) Respond in JSON format!

here is an example in JSON format:
{get_fewshot_sample()[0]['context']}                                              
"""}]
    return messages



def get_fewshot_sample():
    samples = [{"context": """Objects:
('bacon','fridge','agent','stove','tray','pan')
Task:
cook a bacon
Initial Scene Graph:
('bacon', 'inside', 'fridge')
('tray', 'inside', 'fridge')

Goal Scene Graph
('bacon', 'ontop', 'pan')
('pan', 'ontop', 'stove')
---         
{
"Step 1": {"Planning":"Open the fridge", "Target": ["fridge"], "Expect Scene Graph": [["bacon", "inside", "fridge"], ["tray", "inside", "fridge"]]},
"Step 2": {"Planning":"Take the tray out of the fridge", "Target": ["fridge","tray"], "Expect Scene Graph": ["bacon", "inside", "fridge"]},
"Step 3": {"Planning":"Take the bacon out of the tray", "Target": ["fridge","bacon"],  "Expect Scene Graph": null},
"Step 4": {"Planning":"Turn on the stove",  "Target": ["stove"], "Expect Scene Graph": ["bacon", "inside", "fridge"]},
"Step 5": {"Planning":"Put the pan on the stove",  "Target": ["fridge","stove"], "Expect Scene Graph": ["pan", "ontop", "stove"]},
"Step 6": {"Planning":"Put the bacon on the pan", "Target": ["bacon","pan"],  "Expect Scene Graph": [["bacon", "ontop", "pan"],["pan", "ontop", "stove"]]},
"Step 7": {"Planning":"Cook the bacon",  "Target": ["bacon"], "Expect Scene Graph": [["bacon", "ontop", "pan"],["pan", "ontop", "stove"]]}
}
"""}]
    return samples



