

## 🔧 Install SAM-HQ Weights



This project requires the **SAM-HQ (Segment Anything High-Quality)** model.

Please download the official weight file from HuggingFace:

👉 **[https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_l.pth](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_l.pth)**

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

## 🚀 Training on GPU

To start training with GPU acceleration, run:

```bash
python train_gpu.py
```

Make sure your environment has:

* PyTorch with CUDA
* stable-baselines3
* tqdm
* wandb (optional)
* scikit-learn
* PIL
* numpy

Example installation:

```bash
pip install -r requirements.txt
```

---

## 📁 Required Data Structure

Ensure your folders follow this layout:
dataset: https://nturlcoursefa-mrl3467.slack.com/archives/C09PQU9RZND/p1765264796368009

```
prompts/
    pore1_4on4/
        initial_prompts/
            <prefix>_features.pt
            <prefix>_initial_indices_pos.pt
            <prefix>_initial_indices_neg.pt

dataset/
    pore1_4on4/
        pore1_images/
            <prefix>.png
        pore1_masks/
            <prefix>.png
```

Each `<prefix>` must have:

* features
* pos indices
* neg indices
* raw image
* ground truth mask

