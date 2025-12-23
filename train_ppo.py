import os
import torch
import numpy as np
from tqdm import tqdm
import time

# Stable Baselines3 Imports
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv 
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from utils_gpu import TensorGraphEnv
from PIL import Image
from segmenter.segment import loading_seg
from lora import LoRA_sam
from sklearn.decomposition import PCA
from segment_anything_hq import sam_model_registry as sam_hq_registry

# Global Path Constants
# Global Path Constants
SAM_HQ_CHECKPOINT_PATH = "./segmenter/checkpoint/sam_hq_vit_l.pth"
BASE_DIR = './training/prompts'
MASK_DIR = './training/masks'
IMAGE_DIR = './training/images'
RESULTS_BASE_DIR = './results_ppo'
# Training Configuration
RUN_NAME = "251213_useLessPoint"
LOG_DIR_NAME = "ppo_logs"
TENSORBOARD_DIR_NAME = "ppo_tensorboard"
CHECKPOINT_PREFIX = "model"
FINAL_MODEL_NAME = "final_ppo_model"
WANDB_PROJECT = "RL_Final_PPO"


def load_models(device, use_LoRA=True):
    """Load Base SAM (for segmentation) and LoRA SAM Encoder (for features)."""
    print("Loading Base SAM for segmentation...")
    # 1. 載入原始 SAM (用於最終分割)
    model_seg_base = loading_seg('vith', device)
    model_seg_base = model_seg_base.to(device)
    if use_LoRA:
        print("Loading LoRA safetensors")
        # 2. 載入另一個 SAM 並套用 LoRA (用於特徵提取)
        # 我們需要一個新的實例，以免汙染上面的 model_seg_base
        model_for_lora = loading_seg('vitl', device)
        sam_lora = LoRA_sam(model_for_lora, 512)
        sam_lora.load_lora_parameters(f"lora_rank512.safetensors")
        model_seg_lora = sam_lora.sam.to(device)
        return model_seg_lora
    else:
        return model_seg_base
    
def loading_sam_hq(model_type, device):
    if model_type == 'vitl':
        checkpoint_path = SAM_HQ_CHECKPOINT_PATH
        model_key = 'vit_l'
    else:
        raise ValueError(f"不支援的 SAM-HQ 型號: {model_type}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到權重檔: {checkpoint_path}")

    print(f"正在載入 SAM-HQ ({model_type}) ...")
    
    sam = sam_hq_registry[model_key](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    return sam

def normalize_core(core):
    """移除檔名中的副檔名"""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        if core.endswith(ext):
            return core[:-len(ext)]
    return core

class IoULoggingCallback(BaseCallback):
    """
    Record the IoU metric from every step
    """
    def __init__(self, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_dir = os.path.join(save_dir, "train_masks")
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = {
            "iou": [],
            "recall": [],
            # "nnd_std_pos": [],
            # "nnd_std_neg": [],
            "pos_count": [],
            "neg_count": [],
            "feat_cross": []
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        if infos:
            info = infos[0]

            for key in self.history.keys():
                if key in info:
                    self.history[key].append(info[key])

        if dones[0]:
            if self.history["iou"]:
                # self.logger.record("train/mean_episode_iou", np.mean(self.history["iou"]))
                self.logger.record("train/final_episode_iou", self.history["iou"][-1])

            if self.history["recall"]:
                self.logger.record("train/final_recall", self.history["recall"][-1])

            # if self.history["nnd_std_pos"]:
            #     # self.logger.record("train/mean_nnd_std_pos", np.mean(self.history["nnd_std_pos"]))
            #     self.logger.record("train/final_nnd_std_pos", self.history["nnd_std_pos"][-1])

            # if self.history["nnd_std_neg"]:
            #     # self.logger.record("train/mean_nnd_std_neg", np.mean(self.history["nnd_std_neg"]))
            #     self.logger.record("train/final_nnd_std_neg", self.history["nnd_std_neg"][-1])

            if self.history["pos_count"]:
                self.logger.record("train/final_pos_count", self.history["pos_count"][-1])
            
            if self.history["neg_count"]:
                self.logger.record("train/final_neg_count", self.history["neg_count"][-1])
            
            if self.history["feat_cross"]:
                self.logger.record("train/final_feat_cross", self.history["feat_cross"][-1])
            
            if "final_mask" in infos[0]:
                mask_np = infos[0]["final_mask"]
                prefix = infos[0].get("prefix", "unknown")
                steps = infos[0].get("episode_steps", 0)

                filename = f"{prefix}_step{self.num_timesteps}.png"
                save_path = os.path.join(self.save_dir, filename)
                
                try:
                    Image.fromarray(mask_np).save(save_path)
                except Exception as e:
                    print(f"Error saving mask: {e}")
            
            for key in self.history:
                self.history[key] = []
            
        return True

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

            max_points = 5
            if len(pos_indices) > max_points:
                perm = torch.randperm(len(pos_indices), device='cpu')[:max_points]
                pos_indices = pos_indices[perm]
                
            if len(neg_indices) > max_points:
                perm = torch.randperm(len(neg_indices), device='cpu')[:max_points]
                neg_indices = neg_indices[perm]

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

def make_env(preloaded_data, image_size, sam_model, max_steps=100, device='cuda', use_samhq=False, embed_scale=1.0):
    def _init():
        env = TensorGraphEnv(preloaded_data, image_size, sam_model, max_steps=max_steps, device=device, inference_mode=False, use_samhq=use_samhq, embed_scale=embed_scale)
        return Monitor(env)
    return _init

def main():
    # --- 設定 ---
    # [修改點 4] 強制使用 cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    run_name = RUN_NAME
    result_dir = os.path.join(RESULTS_BASE_DIR, run_name)
    os.makedirs(result_dir, exist_ok=True)
    image_size = 140

    base_dir = BASE_DIR
    log_dir = os.path.join(result_dir, LOG_DIR_NAME)
    tensorboard_dir = os.path.join(result_dir, TENSORBOARD_DIR_NAME)
    
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1500000,
        "env_name": "TensorGraphEnv",
        "learning_rate": 3e-4,    
        "n_steps": 2000,          
        "batch_size": 100,         
        "n_epochs": 5,           
        "gamma": 0.95,            
        "gae_lambda": 0.95,       
        "clip_range": 0.2,        
        "ent_coef": 0.01,         
        "vf_coef": 0.5,           
        "max_grad_norm": 0.5,
        "embed_scale": 0.5,
        "use_samhq": True   
    }
    
    run = wandb.init(
        project=WANDB_PROJECT,
        config=config,
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,  
    )

    print("Loading SAM Model...")
    sam_model = loading_sam_hq('vitl', device)
    sam_model.to(device)
    sam_model.eval()
    for param in sam_model.parameters():
        param.requires_grad = False

    # mask_dir = './dataset/original/target_masks'
    mask_dir = MASK_DIR
    image_dir = IMAGE_DIR

    preloaded_data = load_data(base_dir, mask_dir, image_dir)
    preloaded_data, pca_model = apply_pca_reduction(preloaded_data, target_dim=32)

    num_envs = 1    
    env = DummyVecEnv([make_env(preloaded_data, image_size, sam_model, max_steps=500, device='cuda', use_samhq=config["use_samhq"], embed_scale=config["embed_scale"]) for _ in range(num_envs)])
    
    print(f"Environment initialized with {num_envs} process (GPU Accelerated).", flush=True)

    # 3. 建立 DQN 模型
    model = PPO(
        config["policy_type"],
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=1,
        tensorboard_log=tensorboard_dir,
        device=device,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])]),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=log_dir,
        name_prefix=CHECKPOINT_PREFIX
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        verbose=2,
    )

    iou_callback = IoULoggingCallback(save_dir=result_dir)
    
    callback_list = CallbackList([checkpoint_callback, wandb_callback, iou_callback])
    
    print("Start training...", flush=True)
    model.learn(
        total_timesteps=config["total_timesteps"], 
        callback=callback_list
    )

    model.save(os.path.join(result_dir, FINAL_MODEL_NAME))
    print("Training finished. Model saved.", flush=True)
    run.finish()

if __name__ == "__main__":
    main()