"""Implementation derived from https://github.com/tloen/alpaca-lora"""
# import ptvsd
# ptvsd.enable_attach(address=('10.140.0.184', 5678))
# ptvsd.wait_for_attach()

import sys
from pathlib import Path
# support running without installing as a package
import numpy as np

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import torch
import json
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
import os

IGNORE_INDEX = -1

def prepare(
    experiment_name: str = None,
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    max_seq_length: int = 1024,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    
    base_path="data/OGSG_data"
    tokenizer = Tokenizer(tokenizer_path)
    
    experiment_path=os.path.join(base_path,experiment_name)

    train_path=f"{experiment_path}/{experiment_name}_train.json"
    test_path=f"{experiment_path}/{experiment_name}_test.json"
    with open(train_path, "r") as file:
        train_dataset = json.load(file)

    with open(test_path, "r") as file:
        test_dataset = json.load(file)        

    print("Processing train split ...")
    train_set = [prepare_sample(train_dataset[sample], tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_dataset.keys())]

    len_list = []
    prompt_list = []
    cho=[]
    for train_set_one in train_set:
        len_list.append(train_set_one["input_ids"].shape[0])
        if len_list[-1] == max_seq_length:
            prompt_list.append(generate_prompt(train_set_one) + train_set_one["answer"])
            cho.append(train_set_one)
    len_list = np.asarray(len_list)

    print(f"Number of max_seq_lengths greater than {max_seq_length}：", np.sum(len_list >= max_seq_length))
    print("max_seq_length", np.max(len_list))
    print("min_seq_length", np.min(len_list))

    torch.save(train_set, f"{experiment_path}/train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(test_dataset[sample], tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_dataset.keys())]

    torch.save(test_set, f"{experiment_path}/test.pt")

def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["answer"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # You can use the following three lines of code to check if the dataset makes sense
    # encoded_full_prompt = tokenize_without_max(tokenizer, full_prompt, max_length=max_length, eos=False)
    # encoded_full_prompt_and_response = tokenize_without_max(tokenizer, full_prompt_and_response,
    #                                                           eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def tokenize_without_max(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    # Original prompt
    # if example["input"]:
    #     return (
    #         "Below is an instruction that describes a task, paired with an input that provides further context. "
    #         "Write a response that appropriately completes the request.\n\n"
    #         f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    #     )
    prompt=f"### Instruction:\n{example['instruction']}\n### Response:\n"
    return prompt
    # return (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     f"### Instruction:\n{example['goal']}{example['SceneGraph']}\n\n### Response:\n"
    # )

if __name__ == "__main__":
    OPTIONS=["initsg_planning","initsg_targetobj_planning","initsg_finalsg_planning","initsg_finalsg_targetobj_planning"]
    for OPTION in OPTIONS:
        prepare(experiment_name=OPTION)



