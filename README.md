<div align="center">
<img src="figure/logo.png" alt="SG_VLM Logo" width="200"/>

# SG_VLM
</div>

# Scene Graph Enhanced Embodied Task Planning
Official implementation of "Scene Graph Enhanced Embodied Task Planning with Large Language Models".

Try out the web demo 🤗 of **SG_VLM**: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yourusername/SG_VLM)

The repository contains:
- The dataset used for fine-tuning the model.
- Code for generating the dataset.
- Scripts for fine-tuning the model on high-performance GPUs.
- Inference scripts for real-time task execution.

## News
- **[2024.04.12]** Training code for **SG_VLM** has been released. 📌

## Overview

SG_VLM utilizes a scene graph approach to understand and plan tasks in dynamic environments better. The model first constructs a comprehensive scene graph from multi-angle images, identifying relationships and attributes of objects within a scene. This structured data then guides the generation of action plans for robotics tasks, improving accuracy and context-awareness.

<div align="center">
  <img src="./howto/figure/sg_vlm_architecture.png" width="95%"/>
</div>

## Setup
Here's a script to set up **SG_VLM** from scratch.
```bash
# Install dependencies
conda create -n sgvlm python=3.10
conda activate sgvlm
git clone https://github.com/yourusername/SG_VLM.git
cd SG_VLM
pip install -r requirements.txt

# Optional: setup for multi-GPU
pip install deepspeed
```

<details>
<summary> <strong> Troubleshooting installation issues </strong> </summary>

1. Ensure your Python version is compatible.
2. Check network settings if dependencies fail to download.
3. Verify GPU compatibility and drivers.
</details>

```bash
# Additional libraries might be required depending on your specific hardware and software setup.
```

## Data Release
[`dataset.json`](./data/dataset.json) includes the task planning data used for model training. The format is detailed, providing object attributes, spatial relationships, and task-specific action sequences.

## Data Generation Process
The dataset creation is automated as follows:
```bash
cd create_dataset
python create_scene_graphs.py
python create_task_instructions.py
python compile_dataset.py
```

## Fine-tuning
Fine-tuning details:
```bash
# Assuming CUDA and appropriate GPUs are available
cd finetune
python finetune_model.py
```

## Inference
To execute task planning in real-time:
```bash
python run_inference.py --input "path_to_input_image"
```

## Validation and Testing
Validation is crucial to ensure robust model performance:
```bash
cd validate
python validate_tasks.py
```

## Contributing
Contributions to SG_VLM are welcome! Please refer to `CONTRIBUTING.md` for guidelines on how to contribute effectively.

## License
SG_VLM is released under the MIT License. See `LICENSE` for more information.

---

Feel free to adjust the content to better fit your project specifics, such as adding more details about the dataset, installation procedures, and any dependencies or submodules your project might have.