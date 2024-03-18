"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
import numpy as np

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import re
import torch
import requests
import json
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
import math
import random



IGNORE_INDEX = -1

def split_dataset(dataset,train_ratio):
    train_set={}
    test_set={}
    remove_repeat=lambda x:list(set(x))
    all_task=[]
    for ele in dataset.keys():
        task=''
        for name in ele.split("_")[:-3]:
            task+=name+'_'
        all_task.append(task[:-1])
    all_task=remove_repeat(all_task)
    random.shuffle(all_task)
    train_keys=all_task[:math.floor(len(all_task)*train_ratio)]
    test_keys=all_task[math.floor(len(all_task)*train_ratio):]
    for ele in dataset.keys():
        for train_key in train_keys:
            if train_key in ele:
                train_set[ele]=dataset[ele]
        for test_key in test_keys:
            if test_key in ele:
                test_set[ele]=dataset[ele]

    return train_set,test_set


DATA_FILE_NAME = "Octopus_iterative_executable_planning.json"
def prepare(
    destination_path: Path = Path("data/Octopus/Octopus_iterative_executable_planning"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    train_ratio=1,
    test_split_size: int = 600,
    max_seq_length: int = 1024,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = DATA_FILE_NAME
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / data_file_name
    # download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    with open(file_path, "r") as file:
        data = json.load(file)

    train_set,test_set=split_dataset(data,train_ratio)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(train_set[sample], tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set.keys())]

    len_list = []
    prompt_list = []
    cho=[]
    for train_set_one in train_set:
        len_list.append(train_set_one["input_ids"].shape[0])
        if len_list[-1] == max_seq_length:
            prompt_list.append(generate_prompt(train_set_one) + train_set_one["answer"])
            cho.append(train_set_one)
    len_list = np.asarray(len_list)

    # If the number greater than 512 is too large,
    # you may need to increase the value of max_seq_length or modify the dataset
    print(f"Number of max_seq_lengths greater than {max_seq_length}：", np.sum(len_list >= max_seq_length))
    print("max_seq_length", np.max(len_list))
    print("min_seq_length", np.min(len_list))

    torch.save(train_set, file_path.parent / "train_15k.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(test_set[sample], tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set.keys())]

    torch.save(test_set, file_path.parent / "test_15k.pt")

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
    prompt=f"### Instruction:\n{example['instruction']}### Response:\n"
    return prompt
    # return (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     f"### Instruction:\n{example['goal']}{example['SceneGraph']}\n\n### Response:\n"
    # )

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)



