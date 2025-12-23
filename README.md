
# Data-Efficient Optimization of the Segment Anything Model for Automatic Bubble Segmentation



## Brief Introduction

This project proposes a **data-efficient reinforcement learning framework** to optimize the **Segment Anything Model (SAM)** for automatic bubble (pore) segmentation.
Instead of relying on manually designed point prompts, we formulate prompt placement as a sequential decision-making problem and train an RL agent to automatically select informative point prompts that improve segmentation quality.
The SAM backbone remains frozen, enabling efficient adaptation with limited proprietary data while maintaining strong generalization.

---

## Environments (Mandatory)

Please ensure the following environment setup:

```bash
conda activate <your_env_name>
```

* Python **3.9.20**
* Do `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117` first
* Dependencies listed in `requirements.txt`

---

## �� Install SAM-HQ Weights

This project requires the **SAM-HQ (Segment Anything High-Quality)** model.

Download the official weights from HuggingFace:

👉 [https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_l.pth](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_l.pth)

After downloading, place the file at:

```
segmenter/checkpoint/sam_hq_vit_l.pth
```

Final directory structure:

```
segmenter/
    checkpoint/
        sam_hq_vit_l.pth
```

If the file is missing, training will raise:

```
FileNotFoundError: ./segmenter/checkpoint/sam_hq_vit_l.pth not found
```

---

## Execution Overview

* Training **must be performed on GPU**
* CUDA-enabled PyTorch is required


---

## Proprietary Data Notice

The dataset and the file `lora_rank512.safetensors` used in this work are **proprietary** and **not publicly available** (provided by 材料所).
Therefore, we cannot release these files publicly.

For users who have access to the dataset, the data should be organized into **separate training and testing splits**, each containing three subdirectories: prompts, masks, and images.
The expected directory structure is as follows:

```text
training/
├── prompts/
├── masks/
└── images/

testing/
├── prompts/
├── masks/
└── images/
```

where:

* `prompts/` contains pre-generated prompt files (e.g., initial point indices or feature-based prompts),
* `masks/` contains the corresponding ground-truth segmentation masks,
* `images/` contains the raw input images.


To facilitate reproducibility and evaluation, we instead provide **pretrained model weights** that can be used for inference and benchmarking without access to the proprietary data.

###  Pretrained Model
the model has been incude in zip 

```
results_ppo/251213_ppo/final_ppo_model.zip
```

---

## Dependencies

Make sure your environment includes:

* PyTorch (with CUDA)
* stable-baselines3
* tqdm
* wandb (optional)
* scikit-learn
* PIL
* numpy

---

## Training (Mandatory)

### Step 1: Generate Initial Prompts (Optional)

If **initial prompts do not exist**, run:

```bash
python Generate_initial_prompts.py
```

If initial prompts already exist, **this step can be skipped**.

---

### Step 2: Train the RL Agent

```bash
python train.py
```

This step trains the RL agent to optimize point prompt placement for SAM.

---

## Inference

To generate segmentation masks for the testing set:

```bash
python inference.py
```

The script outputs predicted mask images for downstream evaluation.



## Evaluation

To compute quantitative metrics:

```bash
python eval.py
```

This script compares predicted masks with ground truth and reports segmentation performance.

---

🔗 **Project Repository (code and experimental details):**  
https://github.com/glendawei/RL_Final

---
