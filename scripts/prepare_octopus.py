import json
import re


def parse_planning(instructions):
    return instructions.split("Original Subtasks:")[1].split('Previous')[0].lstrip(" ").lstrip("\n")

def parse_relation(text):
    pattern = r"\('([^']+)', '([^']+)', '([^']+)'\)"

    matches = re.findall(pattern, text)

    for match in matches:
        print(match)


if __name__ == "__main__":
    with open("/mnt/petrelfs/liushuai1/TaPA/data/OctoGibson.json","r")as f:
        octopus_data=json.load(f)

    new_json={}
    for k in octopus_data['data'].keys():
        new_json[k]={}
        one_data=octopus_data['data'][k]
        new_json[k]['objects']=one_data['objects']
        new_json[k]['relations']=one_data['relations']
        new_json[k]['planning']=parse_planning(one_data['instruction'])
        new_json[k]['instruction']=one_data['instruction']
        new_json[k]['answer']=one_data['answer']