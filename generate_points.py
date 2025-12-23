import numpy as np
import torch
import torchvision.transforms as T
# from . import hubconf
import torch.nn.functional as F
from segmenter.segment_anything import sam_model_registry, build_sam_vit_l
from safetensors.torch import load_file
from lora import LoRA_sam


# generate_points.py

import cv2
from segmenter.segment import SamPredictor # 需要這個來跑 LoRA 推論

def get_error_regions(pred_mask, gt_mask):
    """找出 False Negative (漏抓) 和 False Positive (誤抓)"""
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    false_neg = np.logical_and(gt_bin == 1, pred_bin == 0)
    false_pos = np.logical_and(gt_bin == 0, pred_bin == 1)
    
    return false_neg, false_pos

def sample_indices_from_mask(mask, size, num_points=20):
    """
    Sample from mask, and transform back to match size
    """
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)

    if eroded_mask.sum() == 0:
        eroded_mask = mask_uint8
    
    y_indices, x_indices = np.where(eroded_mask > 0)
    
    if len(y_indices) == 0:
        return torch.tensor([], dtype=torch.long)

    count = min(len(y_indices), num_points)
    sampled_idx = np.random.choice(len(y_indices), size=count, replace=False)
    
    patch_rows = y_indices[sampled_idx] // 14
    patch_cols = x_indices[sampled_idx] // 14
    
    grid_w = size // 14
    flat_indices = patch_rows * grid_w + patch_cols

    flat_indices = np.unique(flat_indices)
    
    return torch.tensor(flat_indices, dtype=torch.long)

def generate_error_guided(gt_mask_pil, image_pil, device, sam_model, size):
    """
    基於 LoRA 錯誤的初始點生成策略
    """
    predictor = SamPredictor(sam_model)

    image_np = np.array(image_pil)
    predictor.set_image(image_np) 

    box = [0, 0, 560, 560]
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=np.array(box), multimask_output=False)
    initial_mask = masks[0].astype(np.uint8)

    gt_mask = np.array(gt_mask_pil)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    fn_mask, fp_mask = get_error_regions(initial_mask, gt_mask)

    num_points = 200
    pos_indices = sample_indices_from_mask(fn_mask, size, num_points=num_points)
    neg_indices = sample_indices_from_mask(fp_mask, size, num_points=num_points)

    image_encoder = sam_model.image_encoder

    image_inner = [image_pil, image_pil]
    dummy_idx = np.array([[0]])
    _, features, _ = forward_matching_sam_lora(image_inner, dummy_idx, device, image_encoder, size)
    
    return features, pos_indices.to(device), neg_indices.to(device)

def load_lora_sam_encoder(base_checkpoint_path, device='cuda'):
    """
    載入 Base SAM 模型並掛載 LoRA 權重
    """
    print(f"Loading Base SAM: {base_checkpoint_path}...")
    # 1. 載入原始 SAM
    sam = build_sam_vit_l(checkpoint=base_checkpoint_path)
    
    # 2. 載入 LoRA 權重
    # 假設 lora_checkpoint_path 是一個 .pt 或 .pth 檔，裡面存的是 state_dict
    rank = 512
    print(f"Loading LoRA weights: rank {rank}...")
    sam_lora = LoRA_sam(sam, rank)
    sam_lora.load_lora_parameters(f"lora_rank{rank}.safetensors")
    model = sam_lora.sam
    # model = sam
    
    # 3. 套用 LoRA 權重
    # strict=False 是關鍵：它允許我們只更新部分層 (LoRA layers)，忽略沒有變動的層
    # msg = sam.load_state_dict(lora_state_dict, strict=False)
    # print(f"LoRA weights loaded. Missing keys (expected for LoRA): {len(msg.missing_keys)}")
    
    model.to(device)
    return model.image_encoder

def find_foreground_patches(mask_np, size):
    """
    Find the foreground and background patches in the mask.

    Parameters:
    mask_np (numpy array): The mask image.
    size (int): The size of the image.

    Returns:
    list: Foreground patches.
    numpy array: Foreground indices.
    list: Background patches.
    numpy array: Background indices.
    """
    fore_patchs = []
    back_patchs = []
    
    # 确保 mask_np 是正确的形状 (H, W, 3)
    if mask_np.shape[-1] == 4:  # 如果是 RGBA 格式
        mask_np = mask_np[..., :3]  # 只保留 RGB 通道

    for i in range(0, mask_np.shape[0] - 14, 14):
        for j in range(0, mask_np.shape[1] - 14, 14):
            if np.all(mask_np[i:i + 14, j:j + 14] != [0, 0, 0]):
                fore_patchs.append((i, j))
            if np.all(mask_np[i:i + 14, j:j + 14] == [0, 0, 0]):
                back_patchs.append((i, j))

    fore_index = np.empty((len(fore_patchs), 1), dtype=int)
    back_index = np.empty((len(back_patchs), 1), dtype=int)

    for i in range(len(fore_patchs)):
        row = fore_patchs[i][0] / 14
        col = fore_patchs[i][1] / 14
        fore_index[i] = row * (size / 14) + col

    for i in range(len(back_patchs)):
        row = back_patchs[i][0] / 14
        col = back_patchs[i][1] / 14
        back_index[i] = row * (size / 14) + col

    return fore_patchs, fore_index, back_patchs, back_index

def calculate_center_points(indices, size):
    """
    Calculate the center points of each patch.

    Parameters:
    indices (numpy array): The indices of the patches.
    size (int): The size of the image.

    Returns:
    list: Center points of the patches.
    """
    center_points = []
    indices = indices.cpu().numpy()

    for i in range(len(indices)):
        row_index = indices[i] // (size / 14)
        col_index = indices[i] % (size / 14)
        center_x = col_index * 14 + 14 // 2
        center_y = row_index * 14 + 14 // 2
        center_points.append([center_x, center_y])

    return center_points

def map_to_ori_size(resized_coordinates, original_size, size):
    """
    Map the coordinates back to the original image size.

    Parameters:
    resized_coordinates (list or tuple): The resized coordinates.
    original_size (tuple): The original size of the image.
    size (int): The size of the image.

    Returns:
    list or tuple: The coordinates mapped back to the original size.
    """
    original_height, original_width = original_size
    scale_height = original_height / size
    scale_width = original_width / size

    if isinstance(resized_coordinates, tuple):
        resized_x, resized_y = resized_coordinates
        original_x = resized_x * scale_width
        original_y = resized_y * scale_height
        return original_x, original_y
    elif isinstance(resized_coordinates, list):
        original_coordinates = [[round(x * scale_width), round(y * scale_height)] for x, y in resized_coordinates]
        return original_coordinates
    else:
        raise ValueError("Unsupported input format. Please provide a tuple or list of coordinates.")

def convert_to_rgb(image):
    """
    Convert an image to RGB format if it is in RGBA.

    Parameters:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The converted image.
    """
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

def forward_matching(images_inner, index, device, dino, size):
    """
    Perform forward matching to get features and indices.

    Parameters:
    images_inner (list of PIL.Image): The list of images.
    index (list): The list of indices.
    device (torch.device): The device to run the model on.
    dino (model): The DINO model.
    size (int): The size of the image.

    Returns:
    list of PIL.Image: The list of images.
    torch.Tensor: The features.
    torch.Tensor: The minimum indices.
    """
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    imgs_tensor = torch.stack([transform(convert_to_rgb(img))[:3] for img in images_inner]).to(device)
    with torch.no_grad():
        features_dict = dino.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']
    fore_index = torch.tensor(index)
    fore_index  = fore_index.long()
    distances = torch.cdist(features[0][fore_index].squeeze(1), features[1])
    min_values, min_indices = distances.min(dim=1)

    return images_inner, features, min_indices

def loading_dino(device):
    """
    Load the DINO model.

    Parameters:
    device (torch.device): The device to load the model on.

    Returns:
    model: The DINO model.
    """
    dino = hubconf.dinov2_vitg14()
    dino.to(device)
    return dino

def distance_calculate(features, indices_pos, indices_back, size):
    """
    Calculate distances between features and physical points.

    Parameters:
    features (torch.Tensor): The features.
    indices_pos (torch.Tensor): The positive indices.
    indices_back (torch.Tensor): The background indices.
    size (int): The size of the image.

    Returns:
    tuple: Distances between features and physical points.
    """
    final_pos_points = torch.tensor(calculate_center_points(indices_pos, size))
    final_neg_points = torch.tensor(calculate_center_points(indices_back, size))

    feature_pos_distances = torch.cdist(features[1][indices_pos], features[1][indices_pos])
    feature_cross_distances = torch.cdist(features[1][indices_pos], features[1][indices_back])
    physical_pos_distances = torch.cdist(final_pos_points, final_pos_points)
    physical_cross_distances = torch.cdist(final_pos_points, final_neg_points)

    return feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_cross_distances

def points_generate(indices_pos, indices_neg, size, images_inner):
    """
    Generate points and map them back to the original size.

    Parameters:
    indices_pos (torch.Tensor): The positive indices.
    indices_neg (torch.Tensor): The negative indices.
    size (int): The size of the image.
    images_inner (list of PIL.Image): The list of images.

    Returns:
    tuple: The mapped positive and negative points.
    """
    final_pos_points = calculate_center_points(indices_pos, size)
    final_neg_points = calculate_center_points(indices_neg, size)

    final_pos_points = set(tuple(point) for point in final_pos_points)
    final_neg_points = set(tuple(point) for point in final_neg_points)
    image = images_inner[1]
    final_pos_points_map = map_to_ori_size(list(final_pos_points), [image.size[1], image.size[0]], size)
    final_neg_points_map = map_to_ori_size(list(final_neg_points), [image.size[1], image.size[0]], size)

    return final_pos_points_map, final_neg_points_map

def forward_matching_sam_lora(images_inner, index, device, sam_encoder, ref_size, target_size):
    """
    修改版: 支援 Reference 和 Target 不同尺寸
    ref_size: Reference image size (e.g., 560)
    target_size: Target image size (e.g., 140)
    """
    # --- 1. 預處理 ---
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
    
    input_size = 1024 
    resize_transform = T.Resize((input_size, input_size))
    
    imgs_tensor = []
    for img in images_inner:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        t_img = T.ToTensor()(resize_transform(img)).to(device)
        imgs_tensor.append(t_img)
        
    imgs_tensor = torch.stack(imgs_tensor)
    imgs_tensor = (imgs_tensor * 255.0 - pixel_mean) / pixel_std

    with torch.no_grad():
        # --- 2. 特徵提取 (Common Encoder) ---
        # features_sam shape: (2, 256, 64, 64)
        features_sam = sam_encoder(imgs_tensor)
        
        # --- 3. 分別處理 Reference 和 Target 的特徵網格 ---
        
        # Reference Image (Index 0) -> 使用 ref_size
        # Grid size 例如: 560 // 14 = 40 (40x40)
        grid_ref = ref_size // 14
        feat_ref = features_sam[0:1] # (1, 256, 64, 64)
        feat_ref_interp = F.interpolate(
            feat_ref, 
            size=(grid_ref, grid_ref), 
            mode='bilinear', 
            align_corners=False
        )
        # Flatten: (1, 256, H, W) -> (1, H*W, 256)
        feat_ref_flat = feat_ref_interp.flatten(2).transpose(1, 2)

        # Target Image (Index 1) -> 使用 target_size
        # Grid size 例如: 140 // 14 = 10 (10x10)
        grid_tgt = target_size // 14
        feat_tgt = features_sam[1:2] # (1, 256, 64, 64)
        feat_tgt_interp = F.interpolate(
            feat_tgt, 
            size=(grid_tgt, grid_tgt), 
            mode='bilinear', 
            align_corners=False
        )
        feat_tgt_flat = feat_tgt_interp.flatten(2).transpose(1, 2)
        
    # --- 4. 計算距離 ---
    fore_index = torch.tensor(index).long().to(device)

    # feat_ref_flat[0][fore_index]: 取出 Reference 圖片中前景的特徵向量
    # feat_tgt_flat[0]: Target 圖片的所有特徵向量
    
    distances = torch.cdist(feat_ref_flat[0][fore_index].squeeze(1), feat_tgt_flat[0])
    
    # 找出 Target 中與 Reference 前景最相似的點
    min_values, min_indices = distances.min(dim=1)
    return images_inner, feat_tgt_flat, min_indices

def generate(mask, image_inner, device, model_encoder, ref_size, target_size):
    """
    修改版: 接受 ref_size 和 target_size
    """
    mask = np.array(mask)
    if len(mask.shape) == 2:
        mask_np = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    else:
        mask_np = mask

    # 1. 找出 Reference Mask 的前景/背景 Patch
    # 注意: 這裡必須使用 ref_size，因為 mask 是 reference 的 mask (560x560)
    fore_patchs, fore_index, back_patchs, back_index = find_foreground_patches(mask_np, ref_size)

    # 2. 前景匹配 (Reference -> Target)
    _, features, initial_indices = forward_matching_sam_lora(
        image_inner, fore_index, device, model_encoder, ref_size, target_size
    )
    
    # 3. 背景匹配 (Reference -> Target)
    _, features_back, initial_indices_back = forward_matching_sam_lora(
        image_inner, back_index, device, model_encoder, ref_size, target_size
    )

    return features, initial_indices, initial_indices_back
