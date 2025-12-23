import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import warnings
import matplotlib.pyplot as plt
import json
from utils_test import calculate_center_points

from generate_points import generate, load_lora_sam_encoder, generate_error_guided
from segmenter.segment import show_points, loading_seg
from lora import LoRA_sam


# Ignore all warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Set paths for feature matching and segmentation modules (kept global for reuse)
generate_path = os.path.join(SCRIPT_DIR, 'feature_matching')
segment_path = os.path.join(SCRIPT_DIR, 'segmenter')
sys.path.append(segment_path)
sys.path.append(generate_path)

# Global data/prompts roots
DATASET_NAME = 'pore1_4on4'
SAVE_FOLDER_NAME = f'{DATASET_NAME}_random10'
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', DATASET_NAME)
PROMPTS_DIR = os.path.join(PROJECT_ROOT, 'prompts', SAVE_FOLDER_NAME)
REFERENCE_IMAGE_DIR = os.path.join(DATASET_DIR, 'reference_images')
MASK_DIR = os.path.join(DATASET_DIR, 'reference_masks')
IMAGE_DIR = os.path.join(DATASET_DIR, 'pore1_images')
INITIAL_PROMPTS_DIR = os.path.join(PROMPTS_DIR, 'initial_prompts')
INITIAL_IMAGES_DIR = os.path.join(PROMPTS_DIR, 'initial_images')
LORA_CHECKPOINT = os.path.join(PROJECT_ROOT, 'lora_rank512.safetensors')

# Function to draw points on an image
def draw_points_on_image(image, points, color):
    """
    Draws a list of points on an image.

    Parameters:
    image (np.array): The image on which to draw the points.
    points (list of tuples): The list of points to draw.
    color (tuple): The color of the points in BGR format.
    """
    image = np.array(image)
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=5, color=color, thickness=-1)
    return image

# Function to save a PyTorch tensor to a text file
def save_tensor_to_txt(tensor, filename):
    """
    Saves a PyTorch tensor to a text file.
    Args:
        tensor (torch.Tensor): The tensor to save.
        filename (str): The path to the text file.
    """
    array = tensor.cpu().numpy()
    np.savetxt(filename, array, fmt='%d')
    print(f"Tensor saved to {filename}")

def calculate_bounding_box(points, patch_size=14, image_size=560):
    """
    Calculate the bounding box that encloses all points with appropriate padding
    
    Parameters:
    points (list): List of points in [x, y] format
    patch_size (int): Size of the feature patch
    image_size (int): Size of the image
    
    Returns:
    tuple: (min_x, min_y, max_x, max_y)
    """
    if not points:
        return None
    
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    # Extend bounding box by 1.5 times the patch size
    padding = int(patch_size * 1.5)
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image_size, max_x + padding)
    max_y = min(image_size, max_y + padding)
    
    return min_x, min_y, max_x, max_y

def main():
    # Hyperparameter setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    ref_image_size = 560
    target_image_size = 140

    model_base = loading_seg('vitl', device)
    print("Loading LoRA SAM Full Model...")
    sam_lora = LoRA_sam(model_base, 512)
    sam_lora.load_lora_parameters(LORA_CHECKPOINT)
    sam_model = sam_lora.sam
    sam_model.to(device)

    # Define directories
    image_prompt_dir = REFERENCE_IMAGE_DIR
    mask_path = MASK_DIR
    image_dir = IMAGE_DIR
    save_dir = INITIAL_PROMPTS_DIR
    initial_image_dir = INITIAL_IMAGES_DIR

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(initial_image_dir, exist_ok=True)
    
    ##
    # Get the reference image for prompting
    reference_list = os.listdir(image_prompt_dir)
    if len(reference_list) == 0:
        raise FileNotFoundError(f"No reference images in {image_prompt_dir}")
    reference = reference_list[0]
    ##

    # Load the reference image (force RGB) and ground truth mask (force L / single channel)
    image_prompt = Image.open(os.path.join(image_prompt_dir, reference)).convert('RGB')
    gt_mask = Image.open(os.path.join(mask_path, reference)).convert('L')


    imglist = os.listdir(image_dir)

    for name in tqdm(imglist):
        image_path = os.path.join(image_dir, name)
        ##
        # force target image to RGB (some SEM images are grayscale)
        image = Image.open(image_path).convert('RGB')
        ##
        
        # target_mask_path = os.path.join(mask_path, name) # 假設檔名相同
        # if not os.path.exists(target_mask_path):
        #     target_mask_path = os.path.join(mask_path, os.path.splitext(name)[0] + '.png')     
        # gt_mask = Image.open(target_mask_path).convert('L').resize((image_size, image_size))

        image_inner = [image_prompt, image]
        features, initial_indices_pos, initial_indices_neg = generate(
            gt_mask, 
            image_inner, 
            device, 
            sam_model.image_encoder,
            ref_size=ref_image_size,
            target_size=target_image_size
        )

        if len(initial_indices_pos) != 0 and len(initial_indices_neg) != 0:
            print("origin: ",len(initial_indices_pos), len(initial_indices_neg))

            initial_indices_pos = torch.unique(initial_indices_pos).to(device)
            initial_indices_neg = torch.unique(initial_indices_neg).to(device)
            print("after remove overlap: ",len(initial_indices_pos), len(initial_indices_neg))
            # Remove intersections
            intersection = set(initial_indices_pos.tolist()).intersection(set(initial_indices_neg.tolist()))
            if intersection:
                initial_indices_pos = torch.tensor([x for x in initial_indices_pos.cpu().tolist() if x not in intersection]).cuda()
                initial_indices_neg = torch.tensor([x for x in initial_indices_neg.cpu().tolist() if x not in intersection]).cuda()
            print("after remove intersection: ",len(initial_indices_pos), len(initial_indices_neg))

            num_samples = 10

            if len(initial_indices_pos) > num_samples:
                perm = torch.randperm(len(initial_indices_pos))
                idx = perm[:num_samples]
                initial_indices_pos = initial_indices_pos[idx]

            if len(initial_indices_neg) > num_samples:
                perm = torch.randperm(len(initial_indices_neg))
                idx = perm[:num_samples]
                initial_indices_neg = initial_indices_neg[idx]
            
            print(f"After sampling (max {num_samples}): Pos={len(initial_indices_pos)}, Neg={len(initial_indices_neg)}")

            core = os.path.splitext(name)[0]
            torch.save(features.cpu(), os.path.join(save_dir, core + '_features.pt'))
            torch.save(initial_indices_pos.cpu(), os.path.join(save_dir, core + '_initial_indices_pos.pt'))
            torch.save(initial_indices_neg.cpu(), os.path.join(save_dir, core + '_initial_indices_neg.pt'))
            
            # Create figure and plot image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            # Prepare points and labels for visualization
            pos_points = calculate_center_points(initial_indices_pos, target_image_size)
            neg_points = calculate_center_points(initial_indices_neg, target_image_size)
            coords = np.array(pos_points + neg_points)
            labels = np.concatenate([
                np.ones(len(initial_indices_pos)),  
                np.zeros(len(initial_indices_neg))
            ])
            show_points(coords, labels, plt.gca())
            
            bbox = calculate_bounding_box(pos_points+neg_points)

            if bbox is not None:
                # 格式: (min_x, min_y, max_x, max_y)
                # 例如：強制框住整張圖的中間
                bbox = (0, 0, target_image_size, target_image_size)

            if bbox is not None:
                min_x, min_y, max_x, max_y = bbox
                bbox_data = {
                    'min_x': int(min_x),
                    'min_y': int(min_y),
                    'max_x': int(max_x),
                    'max_y': int(max_y)
                }
                with open(os.path.join(save_dir, core + '_bbox.json'), 'w') as f:
                    json.dump(bbox_data, f)
                
                rect = plt.Rectangle((min_x, min_y), 
                                  max_x - min_x, 
                                  max_y - min_y,
                                  fill=False,
                                  edgecolor='#f08c00',
                                  linewidth=2.5)
                plt.gca().add_patch(rect)
            
            # Remove axes and save figure
            plt.axis('off')
            plt.savefig(os.path.join(initial_image_dir, f'{core}_initial.png'), bbox_inches='tight', pad_inches=0)
            plt.close()       

        else:
            print(f"No positive or negative indices found for {name}")

if __name__ == "__main__":
    main()
