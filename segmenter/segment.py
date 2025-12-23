import os
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything_hq import SamPredictor as SamHQPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class ScaledSamHQPredictor(SamHQPredictor):
    """
    繼承自 SamHQPredictor，但在將 sparse_embeddings 傳入 decoder 之前，
    會先乘上一個 scaling factor。
    """
    def __init__(self, sam_model, embed_scale=1.0):
        super().__init__(sam_model)
        self.embed_scale = embed_scale  # 儲存您的常數

    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        hq_token_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        覆寫原本的 predict_torch，加入 scaling 邏輯。
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        sparse_embeddings = sparse_embeddings * self.embed_scale

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
            interm_embeddings=self.interm_features,
        )

        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks
    
    def preprocess_prompts(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
    ):
        """
        將 Numpy Prompt 轉換為 SAM 需要的 Tensor，並進行座標縮放。
        回傳的 Tensor 已經準備好可以送入 predict_from_torch_prompts。
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        
        if point_coords is not None:
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            
            # 增加 Batch 維度: (N, 2) -> (1, N, 2)
            # 注意：這裡我們先設 Batch=1，後續 predict_fused_mask 會把它們 concat 起來
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        return coords_torch, labels_torch, box_torch, mask_input_torch

    def predict_from_torch_prompts(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        hq_token_only: bool = False,
    ):
        """
        接收批次化的 Tensor Prompts，執行預測並回傳 Numpy 結果。
        支援 Batch Input: point_coords 可以是 (B, N, 2)
        """
        # 呼叫底層的 predict_torch (它支援 Batch)
        masks, iou_predictions, low_res_masks = self.predict_torch(
            point_coords,
            point_labels,
            boxes,
            mask_input,
            multimask_output,
            return_logits=return_logits,
            hq_token_only=hq_token_only,
        )

        masks_np = masks.detach().cpu().numpy()
        iou_predictions_np = iou_predictions.detach().cpu().numpy()
        low_res_masks_np = low_res_masks.detach().cpu().numpy()
        
        return masks_np, iou_predictions_np, low_res_masks_np

def show_mask(mask, ax, random_color=False):
    """Display mask on the image."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([254 / 255, 215 / 255, 26 / 255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=300):
    """Display points on the image."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='#40e0d0', marker='o', s=marker_size,edgecolor='white',linewidth=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='pink', marker='o', s=marker_size,edgecolor='white',linewidth=1)

def prepare_input(ps_points, ng_points):
    """Prepare input points and labels for the model."""
    if ps_points is not None and ng_points is not None:
        ps_points = np.array(ps_points)
        ng_points = np.array(ng_points)
        input_point = np.vstack((ps_points, ng_points))
        ps_label = np.ones(ps_points.shape[0])
        ng_label = np.zeros(ng_points.shape[0])
        input_label = np.concatenate((ps_label, ng_label))
    else:
        ps_points = np.array(ps_points)
        input_point = ps_points
        input_label = np.ones(ps_points.shape[0])
    return input_point, input_label

def save_max_contour_area(mask):
    """Find and save the largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(mask)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_image, [max_contour], -1, 255, thickness=cv2.FILLED)
    return filled_image

def refine_mask(mask):
    """Refine the mask by keeping only the largest contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    min_area = 0.3 * cv2.contourArea(largest_contour)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)
    return contour_mask

def process_image(image, ps_points, ng_points, sam, max_contour=False, use_samhq=False):
    """Process a single image to generate a mask."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if not use_samhq:
        predictor = SamPredictor(sam)
    else:
        predictor = SamHQPredictor(sam)
    predictor.set_image(image)
    input_point, input_label = prepare_input(ps_points, ng_points)
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    mask_image = (masks[0] * 255).astype(np.uint8)
    if max_contour:
        mask_image = save_max_contour_area(mask_image)
    return mask_image

def loading_seg(model_type, device):
    """Load the segmentation model."""
    if model_type == 'vitb':
        sam_checkpoint = "./segmenter/checkpoint/sam_vit_b_01ec64.pth"
        model_type = 'vit_b'
    elif model_type == 'vitl':
        sam_checkpoint = "./segmenter/checkpoint/sam_vit_l_0b3195.pth"
        model_type = 'vit_l'
    elif model_type == 'vith':
        sam_checkpoint = "./segmenter/checkpoint/sam_vit_h_4b8939.pth"
        model_type = 'vit_h'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

def seg_main(image, pos_prompt, neg_prompt, device, sam_model, max_contour=False, use_samhq=False):
    """Main segmentation function."""
    mask = process_image(image, pos_prompt, neg_prompt, sam_model, max_contour, use_samhq)
    return mask

def get_predictor(sam_model, use_samhq=False, embed_scale=1.0):
    """
    初始化並回傳一個 Predictor 物件。
    """
    if not use_samhq:
        predictor = SamPredictor(sam_model)
    else:
        predictor = ScaledSamHQPredictor(sam_model, embed_scale)
    return predictor

def predict_with_predictor(predictor, pos_points, neg_points):
    """
    使用已經設定好圖片的 predictor 進行快速預測。
    """
    input_point, input_label = prepare_input(pos_points, neg_points)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask_image = (masks[0] * 255).astype(np.uint8)
    return mask_image

def predict_fused_mask(predictor, pos_points, neg_points, image_size, k_neg=3, sigma_ratio=0.1):
    """
    批次化加速版：針對每個正樣本點生成局部 Mask，並進行高斯加權融合。
    如果 Batch 堆疊失敗，會自動退回迴圈處理。
    """
    pos_points = np.array(pos_points)
    neg_points = np.array(neg_points)
    
    h, w = image_size
    num_pos = len(pos_points)
    
    if num_pos == 0:
        return np.zeros((h, w), dtype=np.uint8)

    if len(neg_points) == 0:
        neg_points = np.empty((0, 2))

    # --- 1. 準備 Batch Data ---
    batch_coords_list = []
    batch_labels_list = []

    individual_inputs = []

    for p in pos_points:
        current_neg_points = []
        if len(neg_points) > 0:
            dists = np.linalg.norm(neg_points - p, axis=1)
            if len(dists) > k_neg:
                nearest_indices = np.argsort(dists)[:k_neg]
                current_neg_points = neg_points[nearest_indices]
            else:
                current_neg_points = neg_points
        
        input_point, input_label = prepare_input([p], current_neg_points)

        pt_coords, pt_labels, _, _ = predictor.preprocess_prompts(
            point_coords=input_point,
            point_labels=input_label
        )
        
        batch_coords_list.append(pt_coords)
        batch_labels_list.append(pt_labels)

        individual_inputs.append((pt_coords, pt_labels))

    # --- 2. 嘗試批次推論 ---
    masks = None
    
    try:
        # 嘗試堆疊: (B, N, 2)
        # 如果每個 Prompt 的點數量不同 (N 不同)，這裡會噴 RuntimeError
        batched_coords = torch.cat(batch_coords_list, dim=0)
        batched_labels = torch.cat(batch_labels_list, dim=0)
        
        # 一次性推論
        masks, _, _ = predictor.predict_from_torch_prompts(
            point_coords=batched_coords,
            point_labels=batched_labels,
            multimask_output=False
        )
        
        # masks: (B, 1, H, W) -> 轉 Numpy (B, H, W)
        masks = masks[:, 0, :, :].detach().cpu().numpy()
        
    except RuntimeError:
        # --- [Fallback Logic] 退回迴圈逐一處理 ---
        # 當 tensor 形狀不對齊時執行此段
        masks_list = []
        
        for coords, labels in individual_inputs:
            # 針對單一組資料推論
            # 注意: predict_from_torch_prompts 回傳的是 Numpy
            m, _, _ = predictor.predict_from_torch_prompts(
                point_coords=coords,
                point_labels=labels,
                multimask_output=False
            )
            # m shape: (1, H, W) -> 取出 (H, W)
            masks_list.append(m[0])
            
        # 堆疊成 (B, H, W) 的 Numpy Array
        masks = np.stack(masks_list, axis=0)

    # --- 3. 高斯加權融合 (Vectorized Fusion) ---
    # 此時 masks 已經是一個 (B, H, W) 的 Numpy Array，無論來源是 Batch 還是 Loop
    
    # 建立 Grid (H, W)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    # Pos Centers: (B, 1, 1) 用於廣播
    center_x = pos_points[:, 0].reshape(-1, 1, 1)
    center_y = pos_points[:, 1].reshape(-1, 1, 1)
    
    # 計算距離平方 (B, H, W)
    dist_sq = (grid_x - center_x)**2 + (grid_y - center_y)**2
    
    # 計算權重
    sigma = max(h, w) * sigma_ratio
    weights = np.exp(-dist_sq / (2 * sigma**2))
    
    # 轉為 float 準備加權 (masks 可能是 bool)
    masks_float = masks.astype(np.float32)
    
    weighted_mask_sum = np.sum(masks_float * weights, axis=0)
    weight_sum = np.sum(weights, axis=0)
    
    # 避免除以 0
    final_prob = np.zeros((h, w), dtype=np.float32)
    valid_mask = weight_sum > 1e-5
    final_prob[valid_mask] = weighted_mask_sum[valid_mask] / weight_sum[valid_mask]
    
    final_mask = (final_prob > 0.5).astype(np.uint8) * 255
    
    return final_mask