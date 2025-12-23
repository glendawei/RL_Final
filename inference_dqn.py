import os
import sys
import time
import warnings
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN

segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)

from segmenter.segment import loading_seg, show_points, get_predictor, prepare_input
from utils_gpu import TensorGraphEnv, LocalMaskCache
from utils_test import generate_points
from lora import LoRA_sam
from sklearn.decomposition import PCA
from segment_anything_hq import sam_model_registry as sam_hq_registry

# Ignore all warnings
warnings.filterwarnings("ignore")

SIZE = 140
DATASET = 'pore1_4on4'
IMAGE_SIZE = SIZE
MAX_STEPS = 500

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)

# SAM-HQ Model Path
SAM_HQ_CHECKPOINT_PATH = "./segmenter/checkpoint/sam_hq_vit_l.pth"

# Testing data paths
TEST_PROMPTS_DIR = os.path.join(BASE_DIR, 'testing', 'prompts')
TEST_MASK_DIR = os.path.join(BASE_DIR, 'testing', 'masks')
TEST_IMAGE_DIR = os.path.join(BASE_DIR, 'testing', 'images')

# Directory containing .pt files (use test set)
PROMPTS_DIR = TEST_PROMPTS_DIR

# Directory containing Mask Images (use test set)
MASK_DIR = TEST_MASK_DIR

# Directory containing Raw Images (For visualization only, use test set)
IMAGE_DIR = TEST_IMAGE_DIR

# Result Saving
save_folder_name = "251221_DQN"
RESULTS_DIR = os.path.join(BASE_DIR, 'results_DQN', save_folder_name)
SAVE_DIR = os.path.join(RESULTS_DIR, 'prediction_mask')
MODEL_PATH = os.path.join(BASE_DIR, 'results_DQN', save_folder_name, "final_dqn_model.zip") 

os.makedirs(SAVE_DIR, exist_ok=True)

def load_models(use_LoRA=True):
    """Load Base SAM and LoRA SAM (same as training)"""
    try:
        print("Loading Base SAM for segmentation...")
        model_seg_base = loading_seg('vitl', DEVICE)
        model_seg_base = model_seg_base.to(DEVICE)

        print("Loading LoRA SAM for feature generation...")
        model_for_lora = loading_seg('vitl', DEVICE)
        sam_lora = LoRA_sam(model_for_lora, 512)
        sam_lora.load_lora_parameters(f"lora_rank512.safetensors")
        model_seg_lora = sam_lora.sam.to(DEVICE)
        
        if use_LoRA:
            return model_seg_lora, model_seg_lora.image_encoder
        else:
            return model_seg_base, model_seg_lora.image_encoder
        
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

def loading_sam_hq(model_type, device):
    if model_type == 'vitl':
        checkpoint_path = "./segmenter/checkpoint/sam_hq_vit_l.pth" 
        model_key = 'vit_l'
    else:
        raise ValueError(f"不支援的 SAM-HQ 型號: {model_type}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到權重檔: {checkpoint_path}")

    print(f"正在載入 SAM-HQ ({model_type}) ...")
    
    sam = sam_hq_registry[model_key](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    return sam

def load_data(base_dir, mask_dir, image_dir):
    """Load data as env expected format"""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    all_files = set(os.listdir(base_dir))
    features_suffix = '_features.pt'
    pos_suffix = '_initial_indices_pos.pt'
    neg_suffix = '_initial_indices_neg.pt'
    embed_suffix = '_original_embed.pt'
    
    prefixes = []
    for fn in all_files:
        if fn.endswith(features_suffix):
            core = fn[:-len(features_suffix)]
            candidates_pos = [f for f in all_files if f.endswith(pos_suffix) and f.startswith(core)]
            candidates_neg = [f for f in all_files if f.endswith(neg_suffix) and f.startswith(core)]
            
            if candidates_pos and candidates_neg:
                prefixes.append(core)

    if not prefixes:
        raise RuntimeError(f"No valid data found in {base_dir}")

    print(f"Found {len(prefixes)} samples. Pre-loading data into memory...")
    
    preloaded_data = []
    for prefix in tqdm(prefixes):
        feature_file = os.path.join(base_dir, f"{prefix}{features_suffix}")
        pos_file = os.path.join(base_dir, f"{prefix}{pos_suffix}")
        neg_file = os.path.join(base_dir, f"{prefix}{neg_suffix}")
        embed_file = os.path.join(base_dir, f"{prefix}{embed_suffix}")

        mask_name = prefix + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        image_name = prefix + ".png"
        image_path = os.path.join(image_dir, image_name)
        
        try:
            features = torch.load(feature_file, map_location='cpu')
            pos_indices = torch.load(pos_file, map_location='cpu')
            neg_indices = torch.load(neg_file, map_location='cpu')

            # max_points = 5
            # if len(pos_indices) > max_points:
            #     perm = torch.randperm(len(pos_indices), device='cpu')[:max_points]
            #     pos_indices = pos_indices[perm]
                
            # if len(neg_indices) > max_points:
            #     perm = torch.randperm(len(neg_indices), device='cpu')[:max_points]
            #     neg_indices = neg_indices[perm]

            # embed = torch.load(embed_file, map_location='cpu')

            gt_mask_pil = Image.open(mask_path).convert('L')
            gt_mask = torch.from_numpy(np.array(gt_mask_pil)).float() / 255.0
            gt_mask = (gt_mask > 0.5).float()

            img_pil = Image.open(image_path).convert('RGB')
            img_np = np.array(img_pil)
            
            preloaded_data.append({
            "prefix": prefix,
            "features": features,
            "pos": pos_indices,
            "neg": neg_indices,
            "gt_mask": gt_mask,
            "raw_image": img_np
            })
        except Exception as e:
            print(f"Error loading {prefix}: {e}")
            continue
            
    print(f"Successfully loaded {len(preloaded_data)} samples.")
    return preloaded_data

def apply_pca_reduction(preloaded_data, target_dim=32):
    """
    對所有 preloaded_data 中的 features 進行 PCA 降維
    修正版：自動處理 features 是單一 Tensor 或 Tuple 的情況
    """
    all_feats = []
    sample_rate = 0.1 
    for item in tqdm(preloaded_data, desc="Collecting features for PCA"):
        raw_feat = item['features']    
        if isinstance(raw_feat, (list, tuple)):
            if len(raw_feat) > 1:
                f = raw_feat[1]
            else:
                f = raw_feat[0]
        else:
            if raw_feat.shape[0] == 1:
                f = raw_feat[0]
            else:
                f = raw_feat

        if f.is_sparse:
            f = f.to_dense()

        if f.dim() == 2:
            if f.shape[0] != 1600 and f.shape[1] == 1600:
                f_flat = f.t()
            else:
                f_flat = f
        elif f.dim() == 3:
            f_flat = f.permute(1, 2, 0).reshape(-1, f.shape[0])
        else:
            print(f"Skipping shape: {f.shape}")
            continue

        n_samples = int(f_flat.shape[0] * sample_rate)
        if n_samples > 0:
            indices = torch.randperm(f_flat.shape[0])[:n_samples]
            all_feats.append(f_flat[indices])
    
    if not all_feats:
        raise RuntimeError("No features collected for PCA!")

    training_data = torch.cat(all_feats, dim=0).cpu().numpy()

    print(f"Fitting PCA on {training_data.shape} matrix...")
    pca = PCA(n_components=target_dim)
    pca.fit(training_data)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA Explainable Variance Ratio: {explained_variance:.4f}")

    for item in tqdm(preloaded_data, desc="Transforming features"):
        raw_feat = item['features']
        if isinstance(raw_feat, (list, tuple)):
            if len(raw_feat) > 1:
                f = raw_feat[1]
                ref_feat = raw_feat[0] # 保留 Ref
            else:
                f = raw_feat[0]
                ref_feat = None
        else:
            if raw_feat.shape[0] == 1:
                f = raw_feat[0]
            else:
                f = raw_feat
            ref_feat = None

        if f.is_sparse:
            f = f.to_dense()

        if f.dim() == 2:
            if f.shape[0] != 1600 and f.shape[1] == 1600:
                f_flat = f.t()
            else:
                f_flat = f
        elif f.dim() == 3:
            f_flat = f.permute(1, 2, 0).reshape(-1, f.shape[0])
            
        f_flat_np = f_flat.cpu().numpy()
        f_reduced = pca.transform(f_flat_np)
        f_reduced_tensor = torch.tensor(f_reduced).float()

        item['features'] = (ref_feat, f_reduced_tensor)
        
    return preloaded_data, pca

def generate_fused_mask(predictor, pos_indices, neg_indices, all_coords_norm, image_size):
    """
    模擬 utils_gpu.py 的行為：針對每個 Pos 點，找最近的 Neg 點，生成 Mask 後融合。
    """
    # 1. 準備座標 (正規化 -> 像素)
    # pos_indices/neg_indices 是 Tensor，all_coords_norm 是 Tensor
    if len(pos_indices) == 0:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    pos_indices_np = pos_indices.cpu().numpy()
    neg_indices_np = neg_indices.cpu().numpy()
    all_coords_np = all_coords_norm.cpu().numpy() # Shape: (1600, 2)
    
    pos_points = all_coords_np[pos_indices_np] * image_size
    neg_points = all_coords_np[neg_indices_np] * image_size
    
    # 2. 初始化 Cache (使用與訓練相同的閾值 0.01)
    mask_cache = LocalMaskCache((image_size, image_size), sigma_ratio=0.1, weight_thresh=0.1)
    
    # 3. 逐點預測 (Batch Prompting)
    for i, p in enumerate(pos_points):
        # KNN Logic for Negative Points
        current_neg_points = []
        neg_indices_used = [] 
        
        if len(neg_points) > 0:
            dists = np.linalg.norm(neg_points - p, axis=1)
            k_neg = 3
            if len(dists) > k_neg:
                nearest_indices = np.argsort(dists)[:k_neg]
                current_neg_points = neg_points[nearest_indices]
                # 這裡不需要真的存 neg_indices_used，因為 inference 只做一次 add
            else:
                current_neg_points = neg_points
        else:
            current_neg_points = np.empty((0, 2))
            
        # Predict
        input_point, input_label = prepare_input([p], current_neg_points)
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        local_mask = masks[0].astype(np.float32)
        
        # Add to Cache (這裡用 pos_indices_np[i] 當 key 只是為了不重複，雖然這裡不會有重複)
        mask_cache.add_pos_point(pos_indices_np[i], p, local_mask, [])

    # 4. 取得最終融合 Mask
    return mask_cache.get_final_mask()

def main():
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    agent = DQN.load(MODEL_PATH, device=DEVICE)
    print("Loading SAM Model...")
    sam_model = loading_sam_hq('vitl', DEVICE)
    sam_model.to(DEVICE)
    sam_model.eval()
    for param in sam_model.parameters():
        param.requires_grad = False
    
    predictor = get_predictor(sam_model, use_samhq=True)

    print(f"Loading data from {PROMPTS_DIR}...")
    base_dir = TEST_PROMPTS_DIR 
    mask_dir = TEST_MASK_DIR 
    image_dir = TEST_IMAGE_DIR 

    preloaded_data = load_data(base_dir, mask_dir, image_dir)
    preloaded_data, pca_model = apply_pca_reduction(preloaded_data, target_dim=32)

    print(f"Starting inference on {len(preloaded_data)} images...")
    
    for data_item in tqdm(preloaded_data, desc="Inference"):
        try:
            prefix = data_item['prefix']

            env = TensorGraphEnv(
                [data_item], 
                IMAGE_SIZE, 
                sam_model=None, 
                max_steps=MAX_STEPS, 
                device='cuda', 
                inference_mode=True,
                use_samhq=True,
                fusion_mode="union"
            )

            obs, _ = env.reset()
            done = False
            steps = 0

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1

            final_pos_mask = env.pos_mask
            final_neg_mask = env.neg_mask
            
            opt_pos_indices = torch.nonzero(final_pos_mask, as_tuple=True)[0]
            opt_neg_indices = torch.nonzero(final_neg_mask, as_tuple=True)[0]


            image_name = prefix + ".png"
            image_path = os.path.join(IMAGE_DIR, image_name)
            if not os.path.exists(image_path):
                image_name = prefix + ".jpg"
                image_path = os.path.join(IMAGE_DIR, image_name)
            
            if os.path.exists(image_path):
                image_pil = Image.open(image_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                image_np = np.array(image_pil)
                
                # === 關鍵修改：使用 Batch Prompting + Fusion ===
                predictor.set_image(image_np) # 設定圖片給 SAM
                
                mask_array = generate_fused_mask(
                    predictor, 
                    opt_pos_indices, 
                    opt_neg_indices, 
                    env.all_coords, # 直接用 Env 裡的座標 (Normalized)
                    IMAGE_SIZE
                )

                # Save Mask
                mask_img = Image.fromarray(mask_array)
                mask_img.save(os.path.join(SAVE_DIR, f"{prefix}_mask.png"))

                pos_indices_np = opt_pos_indices.cpu().numpy()
                neg_indices_np = opt_neg_indices.cpu().numpy()
                all_coords_np = env.all_coords.cpu().numpy()
                pos_points = all_coords_np[pos_indices_np] * IMAGE_SIZE
                neg_points = all_coords_np[neg_indices_np] * IMAGE_SIZE

                # Save Prompt Visualization
                plt.figure(figsize=(10, 10))
                plt.imshow(image_pil)
                
                coords = []
                labels = []
                if len(pos_points) > 0:
                    coords.extend(pos_points)
                    labels.extend([1] * len(pos_points))
                if len(neg_points) > 0:
                    coords.extend(neg_points)
                    labels.extend([0] * len(neg_points))
                
                if coords:
                    show_points(np.array(coords), np.array(labels), plt.gca(), marker_size=75)
                
                plt.axis('off')
                plt.savefig(os.path.join(SAVE_DIR, f"{prefix}_prompt.png"))
                plt.close()
            else:
                print(f"Warning: Original image not found for {prefix}, skipping visualization.")

        except Exception as e:
            print(f"Error processing {data_item['prefix']}: {e}")
            continue

    print(f"All done! Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()