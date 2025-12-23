import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
import gc
import warnings
import torch.nn.functional as F
import math

segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
from segmenter.segment_anything import SamPredictor
from segmenter.segment import seg_main, get_predictor, predict_with_predictor, predict_fused_mask, prepare_input
import cv2

warnings.filterwarnings("ignore")

class LocalMaskCache:
    def __init__(self, image_size, mode='gaussian', sigma_ratio=0.1, weight_thresh=0.01):
        """
        A cache that stores the batch prompt mask
        Args:
            image_size: (h, w)
            mode: 'gaussian' (加權平均) 或 'union' (直接疊加)
            sigma_ratio: (僅 Gaussian 模式用) 高斯 Sigma 比例
            weight_thresh: (僅 Gaussian 模式用) 權重截斷閾值
        """
        self.h, self.w = image_size
        self.mode = mode
        self.sigma = max(self.h, self.w) * sigma_ratio
        self.weight_thresh = weight_thresh
        
        self.cache = {}
        self.grid_y, self.grid_x = np.mgrid[0:self.h, 0:self.w]

        # === 根據模式初始化容器 ===
        if self.mode == 'gaussian':
            self.accumulated_prob = np.zeros((self.h, self.w), dtype=np.float32)
            self.accumulated_weight = np.zeros((self.h, self.w), dtype=np.float32)
        elif self.mode == 'union':
            self.vote_count = np.zeros((self.h, self.w), dtype=np.float32)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

    def add_pos_point(self, pos_idx, pos_coord, local_mask, neg_indices_used):
        """新增點 (自動判斷模式)"""
        
        if self.mode == 'gaussian':
            # === Gaussian Mode ===
            dist_sq = (self.grid_x - pos_coord[0])**2 + (self.grid_y - pos_coord[1])**2
            weight_map = np.exp(-dist_sq / (2 * self.sigma**2))
            
            # 有效距離截斷
            if self.weight_thresh is not None:
                weight_map[weight_map < self.weight_thresh] = 0
            
            self.cache[pos_idx] = {
                'mask': local_mask,
                'weight': weight_map, # 存截斷後的權重
                'neg_indices': neg_indices_used
            }
            self.accumulated_prob += local_mask * weight_map
            self.accumulated_weight += weight_map
            
        elif self.mode == 'union':
            # === Union Mode ===
            mask_binary = (local_mask > 0.5).astype(np.float32)
            
            self.cache[pos_idx] = {
                'mask': mask_binary,
                'neg_indices': neg_indices_used
            }
            self.vote_count += mask_binary

    def remove_pos_point(self, pos_idx):
        """移除點"""
        if pos_idx not in self.cache:
            return

        entry = self.cache.pop(pos_idx)
        
        if self.mode == 'gaussian':
            self.accumulated_prob -= entry['mask'] * entry['weight']
            self.accumulated_weight -= entry['weight']
        elif self.mode == 'union':
            self.vote_count -= entry['mask']
            self.vote_count = np.maximum(self.vote_count, 0) # 防止負數

    def get_affected_pos_indices(self, neg_idx, pos_points_dict, k_neg):
        # 這一塊邏輯兩者共用
        affected = []
        for pos_idx, entry in self.cache.items():
            if neg_idx in entry['neg_indices']:
                affected.append(pos_idx)
        return affected
        
    def get_final_mask(self):
        """取得最終 Mask"""
        if self.mode == 'gaussian':
            final_prob = np.zeros((self.h, self.w), dtype=np.float32)
            valid_mask = self.accumulated_weight > 1e-5
            final_prob[valid_mask] = self.accumulated_prob[valid_mask] / self.accumulated_weight[valid_mask]
            return (final_prob > 0.5).astype(np.uint8) * 255
            
        elif self.mode == 'union':
            # 只要票數 >= 1 就是前景
            return (self.vote_count >= 1).astype(np.uint8) * 255
        
class TensorGraphEnv(gym.Env):
    """
    A tensor-based environ without using Networkx
    """
    def __init__(self, preloaded_data, image_size, sam_model, max_steps=100, device='cuda', inference_mode=False, use_samhq=False, embed_scale=1.0, fusion_mode='gaussian', cache_thres=0.1, cache_sigma=0.1):
        super().__init__()
        
        self.preloaded_data = preloaded_data
        self.max_steps = max_steps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.grid_size = 4
        self.patch_side = 14
        self.grid_w = self.image_size // self.patch_side
        self.num_total_patches = self.grid_w * self.grid_w
        self.min_pos_points = 3
        self.min_neg_points = 3
        self.max_pos_points = 15
        self.max_neg_points = 15
        
        # 0: Remove Pos
        # 1: Remove Neg
        # 2: Add Pos
        # 3: Add Neg
        self.action_space = spaces.Discrete(4)

        sample_feat = self.preloaded_data[0]["features"][1] 
        self.feat_dim = sample_feat.shape[-1]
        # obs_dim = 8 + (self.grid_size * self.grid_size) * 2 + (self.feat_dim * 2)
        obs_dim = 8 + (self.feat_dim * 2)
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # === 1. 預先計算並快取所有靜態資料到 GPU ===
        # 座標 (1600, 2)
        all_indices = np.arange(self.num_total_patches)
        coords_np = self._calculate_center_points_numpy(all_indices, image_size)
        
        # 除以 image_size 進行正規化 (0.0 ~ 1.0)
        self.all_coords = torch.tensor(coords_np / image_size, device=self.device, dtype=torch.float32)
        
        # Grid Index 計算邏輯也要微調，因為 coords_np 現在是像素，但 self.all_coords 是正規化值
        # 為了避免混淆，這裡用原本的 coords_np (像素) 來算 Grid Index 比較直觀
        grid_indices_2d = (coords_np / self.image_size * self.grid_size).astype(int)
        grid_indices_2d = np.clip(grid_indices_2d, 0, self.grid_size - 1)
        self.grid_indices_flat = torch.tensor(
            grid_indices_2d[:, 1] * self.grid_size + grid_indices_2d[:, 0], 
            device=self.device, dtype=torch.long
        )

        self.pos_mask = None      
        self.neg_mask = None        
        self.current_features = None
        self.prev_stats = {}

        self.inference_mode = inference_mode
        self.sam_model = sam_model
        self.use_samhq = use_samhq
        self.embed_scale = embed_scale

        # the parameter of LocalCache
        self.fusion_mode = fusion_mode
        self.cache_thres = cache_thres
        self.cache_sigma = cache_sigma

        self.current_gt_mask = None
        self.current_image = None
        self.prev_metric = {}
        self.prev_pred_mask = None

        if not self.inference_mode and self.sam_model is None:
            raise ValueError("sam_model must be provided in training mode")
        
        if not self.inference_mode:
            self.predictor = get_predictor(self.sam_model, use_samhq=self.use_samhq, embed_scale=self.embed_scale)
        
        self.baseline_logits = None
        self.mask_cache = None
        self.prev_pos_indices = set()
        self.prev_neg_indices = set()

    def _calculate_center_points_numpy(self, indices, size):
        """輔助函數：計算座標 (只在 init 跑一次)"""
        patch_dim = size // 14
        rows = indices // patch_dim
        cols = indices % patch_dim
        x = cols * 14 + 7
        y = rows * 14 + 7
        return np.stack([x, y], axis=1)

    def reset(self, seed=None, options=None):
        gc.collect() 
        super().reset(seed=seed)

        data = random.choice(self.preloaded_data)
        self.current_data = data
        
        self.last_pred_mask = None
        self.prev_pred_mask = None

        self.current_features = data["features"][1].to(self.device).float()

        if not self.inference_mode:
            if "gt_mask" not in data:
                raise ValueError("GT Mask missing in training data!")
            self.current_gt_mask = data["gt_mask"].to(self.device)

            if "raw_image" in data:
                self.current_image = data["raw_image"]
                self.predictor.set_image(self.current_image)
            else:
                raise ValueError("Data dictionary missing 'raw_image'. Please update load_data.")
        else:
            self.current_gt_mask = None
            self.current_image = None

        self.pos_mask = torch.zeros(self.num_total_patches, dtype=torch.bool, device=self.device)
        self.neg_mask = torch.zeros(self.num_total_patches, dtype=torch.bool, device=self.device)
        # self.rem_pos_mask = torch.zeros(self.num_total_patches, dtype=torch.bool, device=self.device)
        # self.rem_neg_mask = torch.zeros(self.num_total_patches, dtype=torch.bool, device=self.device)

        pos_idx = data["pos"].to(self.device).long()
        neg_idx = data["neg"].to(self.device).long()

        if len(pos_idx) > self.max_pos_points:
            perm = torch.randperm(len(pos_idx), device=self.device)[:self.max_pos_points]
            pos_idx = pos_idx[perm]
            
        if len(neg_idx) > self.max_neg_points:
            perm = torch.randperm(len(neg_idx), device=self.device)[:self.max_neg_points]
            neg_idx = neg_idx[perm]

        self.pos_mask[pos_idx] = True
        self.neg_mask[neg_idx] = True
        self.neg_mask[self.pos_mask] = False

        if not self.inference_mode:
            # 只有在訓練模式下，環境內部才需要維護 Mask Cache
            self.mask_cache = LocalMaskCache((self.image_size, self.image_size), mode=self.fusion_mode, sigma_ratio=self.cache_sigma, weight_thresh=self.cache_thres)
            
            pos_indices = torch.nonzero(self.pos_mask, as_tuple=True)[0].cpu().numpy()
            neg_indices = torch.nonzero(self.neg_mask, as_tuple=True)[0].cpu().numpy()
            
            self.prev_pos_indices = set(pos_indices)
            self.prev_neg_indices = set(neg_indices)
            self._update_mask_cache_full(pos_indices, neg_indices)
        else:
            # 推論模式下，不需要環境內的 Mask (因為我們會在外部用 generate_fused_mask 生成)
            self.mask_cache = None
            self.prev_pos_indices = set()
            self.prev_neg_indices = set()
        
        self.steps = 0
        self.prev_stats = self._calculate_stats()

        if not self.inference_mode:
            self.prev_metric = self._calculate_reward_metric()
        else:
            self.prev_metric = {}        
        
        return self._get_observation(), {}

    def step(self, action):
        current_pos_count = self.pos_mask.sum().item()
        current_neg_count = self.neg_mask.sum().item()

        self._execute_action_tensor(action)

        invalid_action_penalty = 0.0
        if action == 0 and current_pos_count <= self.min_pos_points:
            invalid_action_penalty = -1.0
        elif action == 1 and current_neg_count <= self.min_neg_points:
            invalid_action_penalty = -1.0
        elif action == 2 and current_pos_count >= self.max_pos_points:
            invalid_action_penalty = -1
        elif action == 3 and current_neg_count >= self.max_neg_points:
            invalid_action_penalty = -1

        curr_stats = self._calculate_stats()
        if not self.inference_mode:
            if self.last_pred_mask is not None:
                self.prev_pred_mask = self.last_pred_mask.clone()
            else:
                self.prev_pred_mask = None
            curr_metric = self._calculate_reward_metric()
            reward = self._compute_reward(curr_stats, self.prev_stats, curr_metric, self.prev_metric)
            reward += invalid_action_penalty
            self.prev_metric = curr_metric
        else:
            curr_metric = {}
            reward = 0.0

        self.prev_stats = curr_stats
        self.steps += 1

        terminated = False
        truncated = self.steps >= self.max_steps

        if not self.inference_mode:
            info = {
                "iou": curr_metric["boundary_iou"],
                "recall": curr_metric["recall"],
                "pos_count": curr_stats['pos_count'],
                "neg_count": curr_stats['neg_count'],
                "feat_cross": curr_stats['feat_cross']
            }
            if (terminated or truncated) and self.last_pred_mask is not None:
                mask_uint8 = (self.last_pred_mask * 255).byte()
                mask_np = mask_uint8.cpu().numpy()
                
                info["final_mask"] = mask_np
                info["prefix"] = self.current_data["prefix"]
                info["episode_steps"] = self.steps
        else:
            info = {}

        return self._get_observation(), reward, terminated, truncated, info
    
    def _update_mask_cache_full(self, pos_indices, neg_indices):
        """全量計算 (用於 Reset 或大幅變動時)"""
        # 這裡其實就是原本 predict_fused_mask 的邏輯，但把結果存進 Cache
        pos_indices = np.array(pos_indices)
        neg_indices = np.array(neg_indices)
        pos_points = (self.all_coords[pos_indices] * self.image_size).cpu().numpy()
        neg_points = (self.all_coords[neg_indices] * self.image_size).cpu().numpy()
        
        # 為了方便查找 Neg，建立 KD-Tree 或直接暴力算
        # 這裡用原本的邏輯
        
        for i, pos_idx in enumerate(pos_indices):
            p = pos_points[i]
            # 找 kNN Neg
            current_neg_points = []
            neg_indices_used = [] # 紀錄用了哪些 Neg 的 Index
            
            if len(neg_points) > 0:
                dists = np.linalg.norm(neg_points - p, axis=1)
                k_neg = 3
                if len(dists) > k_neg:
                    nearest_indices = np.argsort(dists)[:k_neg]
                    current_neg_points = neg_points[nearest_indices]
                    neg_indices_used = neg_indices[nearest_indices]
                else:
                    current_neg_points = neg_points
                    neg_indices_used = neg_indices # 全部
            
            # Predict
            input_point, input_label = prepare_input([p], current_neg_points)
            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            local_mask = masks[0].astype(np.float32)
            
            # 存入 Cache
            self.mask_cache.add_pos_point(pos_idx, p, local_mask, neg_indices_used)
    
    def _get_boundary(self, mask, kernel_size=3):
        """
        利用 Dilation - Erosion 提取邊界。
        mask: (H, W) or (1, H, W) 0/1 Tensor
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
            
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
        boundary = dilated - eroded
        
        return boundary.squeeze()
    
    def _calculate_reward_metric(self):
        # 1. 取得當前索引
        curr_pos_indices = set(torch.nonzero(self.pos_mask, as_tuple=True)[0].cpu().numpy())
        curr_neg_indices = set(torch.nonzero(self.neg_mask, as_tuple=True)[0].cpu().numpy())
        
        # 2. 判斷變動
        added_pos = curr_pos_indices - self.prev_pos_indices
        removed_pos = self.prev_pos_indices - curr_pos_indices
        
        added_neg = curr_neg_indices - self.prev_neg_indices
        removed_neg = self.prev_neg_indices - curr_neg_indices
        
        neg_changed = len(added_neg) > 0 or len(removed_neg) > 0
        
        # 3. 執行增量更新
        
        # A. 移除 Pos: 直接從 Cache 刪掉
        for pos_idx in removed_pos:
            self.mask_cache.remove_pos_point(pos_idx)
            
        # B. 新增 Pos: 計算並加入 Cache
        for pos_idx in added_pos:
            p = (self.all_coords[pos_idx] * self.image_size).cpu().numpy()
            
            # 找 kNN Neg (從當前的 neg_indices 中找)
            # 注意：這裡需要完整的 neg_points 列表
            all_neg_indices = list(curr_neg_indices)
            if len(all_neg_indices) > 0:
                all_neg_points = (self.all_coords[all_neg_indices] * self.image_size).cpu().numpy()
                dists = np.linalg.norm(all_neg_points - p, axis=1)
                k_neg = 3
                if len(dists) > k_neg:
                    nearest_args = np.argsort(dists)[:k_neg]
                    current_neg_points = all_neg_points[nearest_args]
                    neg_indices_used = [all_neg_indices[i] for i in nearest_args]
                else:
                    current_neg_points = all_neg_points
                    neg_indices_used = all_neg_indices
            else:
                current_neg_points = np.empty((0, 2))
                neg_indices_used = []
                
            input_point, input_label = prepare_input([p], current_neg_points)
            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            local_mask = masks[0].astype(np.float32)
            self.mask_cache.add_pos_point(pos_idx, p, local_mask, neg_indices_used)
            
        # C. 負樣本改變 (最複雜的情況)
        # 如果 Neg 變了，原本 Cache 裡的某些 Pos 可能會過期 (因為它們參考的 kNN Neg 變了)
        # 為了簡化，如果 Neg 變了，我們檢查所有的 Pos，看它們是否離變動的 Neg 很近
        if neg_changed:
            # 這一步比較耗時，如果負樣本變動頻繁，這還是會慢
            # 簡單策略：找出所有「kNN 列表中包含被移除 Neg」的 Pos -> 重算
            # 以及「新加入的 Neg 比原本的 kNN 更近」的 Pos -> 重算
            
            # 這裡為了保證正確性且不過度複雜，建議：
            # 如果 Neg 變動，重新計算 *所有* Pos 的 Mask (退回全量更新)
            # 或者：只重新計算那些距離變動 Neg 小於某個閾值的 Pos
            
            # 這裡示範「全量重算 Pos」，因為在只有 3~15 個 Pos 的情況下，這比維護複雜的依賴關係還快
            self.mask_cache = LocalMaskCache((self.image_size, self.image_size), mode=self.fusion_mode, sigma_ratio=self.cache_sigma, weight_thresh=self.cache_thres)
            self._update_mask_cache_full(list(curr_pos_indices), list(curr_neg_indices))

        # 4. 取得最終 Mask
        mask_np = self.mask_cache.get_final_mask()
        
        # 更新狀態
        self.prev_pos_indices = curr_pos_indices
        self.prev_neg_indices = curr_neg_indices
        
        # 轉 Tensor 算 Reward
        mask_tensor = torch.from_numpy(mask_np).to(self.device).float() / 255.0
        pred_mask = (mask_tensor > 0.5).float()

        metric = self._calculate_all_metrics(pred_mask, self.current_gt_mask)
        self.last_pred_mask = pred_mask
        
        return metric
    
    def _morphology(self, mask, kernel_size, method='dilate'):
        """
        使用 MaxPool2d 模擬 Dilation (膨脹) 與 Erosion (腐蝕)
        mask: (H, W) Float Tensor (0.0 or 1.0)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        padding = kernel_size // 2
        
        if method == 'dilate':
            out = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        elif method == 'erode':
            out = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return out.squeeze()

    def _calculate_iou_torch(self, pred_mask, gt_mask):
        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        
        if union == 0: return 0.0
        return (intersection / union).item()

    def _calculate_precision_recall_torch(self, pred_mask, gt_mask):
        tp = (pred_mask & gt_mask).sum().float()
        fp = (pred_mask & ~gt_mask).sum().float()
        fn = (~pred_mask & gt_mask).sum().float()

        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def _calculate_biou_torch(self, pred_mask, gt_mask, dilation_ratio=0.02):
        pred_float = pred_mask.float()
        gt_float = gt_mask.float()
        
        h, w = pred_mask.shape
        diagonal = math.sqrt(h**2 + w**2)
        dilation_pixels = int(diagonal * dilation_ratio)
        if dilation_pixels < 1: dilation_pixels = 1

        k_size = 2 * dilation_pixels + 1
        
        def get_boundary(mask_f):
            eroded = self._morphology(mask_f, kernel_size=3, method='erode')
            return mask_f - eroded

        gt_boundary = get_boundary(gt_float)
        pred_boundary = get_boundary(pred_float)

        gt_dilated = self._morphology(gt_boundary, kernel_size=k_size, method='dilate')
        pred_dilated = self._morphology(pred_boundary, kernel_size=k_size, method='dilate')

        gt_d_bool = gt_dilated > 0.5
        pred_d_bool = pred_dilated > 0.5
        
        intersection = (gt_d_bool & pred_d_bool).sum().float()
        union = (gt_d_bool | pred_d_bool).sum().float()
        
        if union == 0: return 0.0
        return (intersection / union).item()
    
    def _calculate_all_metrics(self, pred, gt):
        if pred.is_floating_point():
            pred_mask = pred > 0.5
        else:
            pred_mask = pred.bool()
            
        if gt.is_floating_point():
            gt_mask = gt > 0.5
        else:
            gt_mask = gt.bool()

        mask_iou = self._calculate_iou_torch(pred_mask, gt_mask)
        precision, recall = self._calculate_precision_recall_torch(pred_mask, gt_mask)
        boundary_iou = self._calculate_biou_torch(pred_mask, gt_mask)

        return {
            "mask_iou": mask_iou,
            "boundary_iou": boundary_iou,
            "precision": precision,
            "recall": recall
        }

    def _execute_action_tensor(self, action):
        pos_indices = torch.nonzero(self.pos_mask, as_tuple=True)[0]
        neg_indices = torch.nonzero(self.neg_mask, as_tuple=True)[0]
        
        # Action 0: Remove Pos
        if action == 0 and len(pos_indices) > self.min_pos_points:
            rand_idx = torch.randint(0, len(pos_indices), (1,), device=self.device).item()
            idx_to_remove = pos_indices[rand_idx]
            self.pos_mask[idx_to_remove] = False
            
        # Action 1: Remove Neg
        elif action == 1 and len(neg_indices) > self.min_neg_points:
            rand_idx = torch.randint(0, len(neg_indices), (1,), device=self.device).item()
            idx_to_remove = neg_indices[rand_idx]
            self.neg_mask[idx_to_remove] = False
            
        # Action 2: Add Pos
        elif action == 2 and len(pos_indices) < self.max_pos_points:
            occupied_mask = self.pos_mask | self.neg_mask
            empty_mask = ~occupied_mask
            empty_indices = torch.nonzero(empty_mask, as_tuple=True)[0]
            
            if len(empty_indices) > 0:
                rand_idx = torch.randint(0, len(empty_indices), (1,), device=self.device).item()
                idx_to_add = empty_indices[rand_idx]
                self.pos_mask[idx_to_add] = True
                
        # Action 3: Add Neg
        elif action == 3 and len(neg_indices) < self.max_neg_points:
            occupied_mask = self.pos_mask | self.neg_mask
            empty_mask = ~occupied_mask
            empty_indices = torch.nonzero(empty_mask, as_tuple=True)[0]
            
            if len(empty_indices) > 0:
                rand_idx = torch.randint(0, len(empty_indices), (1,), device=self.device).item()
                idx_to_add = empty_indices[rand_idx]
                self.neg_mask[idx_to_add] = True

    def _calculate_stats(self):
        stats = {}
        p_feats = self.current_features[self.pos_mask]
        n_feats = self.current_features[self.neg_mask]
        p_coords = self.all_coords[self.pos_mask]
        n_coords = self.all_coords[self.neg_mask]
        
        stats['pos_count'] = len(p_feats)
        stats['neg_count'] = len(n_feats)

        def calc_mean_dist(t1, t2, is_self=False):
            if len(t1) == 0 or len(t2) == 0: return -10
            # cdist shape: (N, M)
            d = torch.cdist(t1, t2)
            
            if is_self:
                if len(t1) <= 1: 
                    return 0.0
                rows, cols = torch.triu_indices(len(t1), len(t2), offset=1)
                valid_dists = d[rows, cols]
                if len(valid_dists) == 0: 
                    return 0.0
                return valid_dists.mean().item()
            
            return d.mean().item()

        stats['feat_pos'] = calc_mean_dist(p_feats, p_feats, is_self=True)
        stats['feat_neg'] = calc_mean_dist(n_feats, n_feats, is_self=True)
        stats['feat_cross'] = calc_mean_dist(p_feats, n_feats)
        stats['phys_pos'] = calc_mean_dist(p_coords, p_coords, is_self=True)
        stats['phys_neg'] = calc_mean_dist(n_coords, n_coords, is_self=True)
        stats['phys_cross'] = calc_mean_dist(p_coords, n_coords)
            
        return stats

    def _get_observation(self):            
        s = self.prev_stats
        base_obs = torch.tensor([
            s['feat_pos'], s['feat_neg'], s['feat_cross'],
            s['phys_pos'], s['phys_neg'], s['phys_cross'],
            s['pos_count'] / self.num_total_patches, # Normalize
            s['neg_count'] / self.num_total_patches, # Normalize
        ], device=self.device, dtype=torch.float32)
        
        p_feats = self.current_features[self.pos_mask] # Shape: (N_pos, feat_dim)
        n_feats = self.current_features[self.neg_mask] # Shape: (N_neg, feat_dim)

        if len(p_feats) > 0:
            p_feat_mean = p_feats.mean(dim=0)
        else:
            p_feat_mean = torch.zeros(self.feat_dim, device=self.device)

        if len(n_feats) > 0:
            n_feat_mean = n_feats.mean(dim=0)
        else:
            n_feat_mean = torch.zeros(self.feat_dim, device=self.device)

        final_obs = torch.cat([base_obs, p_feat_mean, n_feat_mean])

        return final_obs.cpu().numpy()

    def _compute_reward(self, curr, prev, curr_metric, prev_metric):
        """計算 Reward"""
        reward = 0.0
            
        # 3. 特徵差異獎勵 (Feature Cross 越大越好)
        if curr['feat_cross'] > prev['feat_cross']:
            reward += 5.0 * (curr['feat_cross'] - prev['feat_cross'])
        else:
            reward -= 5.0 * (prev['feat_cross'] - curr['feat_cross'])

        if self.current_gt_mask is not None:
            pos_indices = torch.nonzero(self.pos_mask, as_tuple=True)[0]
            neg_indices = torch.nonzero(self.neg_mask, as_tuple=True)[0]

            pos_px = (self.all_coords[pos_indices] * self.image_size).long()
            neg_px = (self.all_coords[neg_indices] * self.image_size).long()

            pos_px = torch.clamp(pos_px, 0, self.image_size - 1)
            neg_px = torch.clamp(neg_px, 0, self.image_size - 1)

            gt_at_pos = self.current_gt_mask[pos_px[:, 1], pos_px[:, 0]]
            gt_at_neg = self.current_gt_mask[neg_px[:, 1], neg_px[:, 0]]

            pos_hits = gt_at_pos.sum().item()
            neg_hits = (1 - gt_at_neg).sum().item()

            pos_misses = len(pos_indices) - pos_hits
            neg_misses = len(neg_indices) - neg_hits

            click_reward = (pos_hits + neg_hits) * 0.5 - (pos_misses + neg_misses) * 0.5
            reward += click_reward

        if self.current_gt_mask is not None and self.prev_pred_mask is not None and self.last_pred_mask is not None:
            gt = self.current_gt_mask > 0.5
            prev_mask = self.prev_pred_mask > 0.5
            curr_mask = self.last_pred_mask > 0.5

            added = curr_mask & (~prev_mask)
            removed = (~curr_mask) & prev_mask

            added_good   = (added & gt).sum().float()
            added_bad    = (added & (~gt)).sum().float()
            removed_good = (removed & gt).sum().float()
            removed_bad  = (removed & (~gt)).sum().float()

            w_add = 0.05
            w_remove = 0.05
            local_reward = (
                w_add * (added_good - added_bad) +
                w_remove * (removed_bad - removed_good)
            )
            reward += local_reward
            if not (added.any().item() or removed.any().item()):
                reward -= 3.0

        reward -= 2
        if isinstance(reward, torch.Tensor):
            reward = reward.item()

        reward_scale = 1.0 / 8.0
        reward = float(np.clip(reward * reward_scale, -10.0, 10.0))
            
        return reward