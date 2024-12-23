import argparse
import os
import copy
import glob
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
from tqdm import tqdm
import json
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
import pandas as pd
import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

from huggingface_hub import hf_hub_download

import json
import glob
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import io, color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import random
from skimage.draw import polygon


from skimage import measure, morphology, color
from skimage.morphology import extrema
from skimage.measure import label
from skimage.filters import threshold_multiotsu, threshold_sauvola
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
# 결과를 원본 해상도로 복원
from skimage.transform import resize

# Hugging Face 허브에서 모델을 로드하는 함수
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda'):
	cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

	args = SLConfig.fromfile(cache_config_file)
	model = build_model(args)
	args.device = device

	cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
	checkpoint = torch.load(cache_file, map_location='cuda')
	log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
	print("Model loaded from {} \n => {}".format(cache_file, log))
	_ = model.eval()
	return model

# 기본 모델들을 로드하는 함수
def load_foundation_models(ckpt_repo_id = "ShilongLiu/GroundingDINO",
                           ckpt_filenmae = "groundingdino_swinb_cogcoor.pth",
                           ckpt_config_filename = "GroundingDINO_SwinB.cfg.py",
                           DEVICE = 'cuda',
                           sam_checkpoint = './weights/sam_vit_h_4b8939.pth'):
    grounding_dino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device=DEVICE)
    if not os.path.exists(sam_checkpoint):
        raise Exception
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    return grounding_dino_model, sam_predictor

# COCO 데이터셋에서 이미지 경로를 반환하는 함수
def image_paths_coco_return(root_dir, coco_json_dir):
	IMAGE_PATHS = sorted(glob.glob(root_dir + '/*.jpg'))
	file_name_to_path = {path.split('/')[-1]: path for path in IMAGE_PATHS}
	coco = COCO(coco_json_dir)

	return file_name_to_path, coco

# Grounding DINO를 사용하여 이미지에서 Bounding Box를 찾는 함수
def DINO_BBOX(image_path, grounding_dino_model, TEXT_PROMPT = "Steel Expansion Joint", BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.25, DEVICE = "cuda"):
	image_source, image_tensor = load_image(image_path)

	boxes, logits, phrases = predict(
		model = grounding_dino_model,
		image = image_tensor,
		caption = TEXT_PROMPT,
		box_threshold = BOX_TRESHOLD,
		text_threshold = TEXT_TRESHOLD,
		device = DEVICE
	)

	if len(boxes) == 0:
		print(f"Failed to find bounding boxes")
		return None, image_source, image_tensor
	elif len(boxes) == 1:
		final_box = boxes[0]
	else:
		areas = [box[2] * box[3] for box in boxes]
		smallest_box = boxes[areas.index(min(areas))]
		final_box = smallest_box

	return final_box, image_source, image_tensor
# COCO 데이터셋에서 주어진 이미지 ID에 대한 어노테이션을 로드하는 함수
def load_annotation(coco, img_id, img):
    img_info = coco.loadImgs(img_id)[0]
    width, height = img_info['width'], img_info['height']

    mask_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # 각 카테고리에 대한 회색조 값 정의
    grayscale_map = {
        26: 128,  # joint
        27: 255,  # gap
    }

    ann_ids = coco.getAnnIds(imgIds = [img_id])
    annotations  = coco.loadAnns(ann_ids)

    for annotation in annotations:
        category_id = annotation['category_id']
        category = coco.loadCats(category_id)[0]['name']
        grayscale_value = grayscale_map.get(category_id, 255) # 기본값 255
        if 'segmentation' in annotation:
            segmentation = annotation['segmentation']
            if isinstance(segmentation, list):
                for seg in segmentation:
                    poly = np.array(seg).reshape((len(seg) // 2, 2))
                    # 다각형 좌표가 이미지 경계를 넘지 않도록 조정
                    poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
                    rr, cc = polygon(poly[:,1], poly[:,0])
                    mask_image[rr, cc] = grayscale_value
    return mask_image


# 주어진 Bounding Box를 기준으로 이미지를 자르는 함수
def bbox_crop_image(box_xyxy, image_path, coco, img_id):
	img = Image.open(image_path)
	img_np = np.array(img)
	mask_image = load_annotation(coco, img_id, img_np)
	mask_image_pil = Image.fromarray(mask_image)

	# Bounding Box를 정수형 좌표로 변환
	left, top, right, bottom = box_xyxy.int().tolist()

	# 이미지와 마스크를 자르기
	cropped_image = img.crop((left, top, right, bottom))
	cropped_mask = mask_image_pil.crop((left, top, right, bottom))

	# numpy 배열로 변환
	cropped_image_np = np.array(cropped_image)
	cropped_mask_np = np.array(cropped_mask)
	return cropped_image_np, cropped_mask_np
# Grounding DINO를 사용하여 이미지에서 잘린 부분을 반환하는 함수
def DINO_CROP(image_source, final_box, coco, img_id):
    H, W, _ = image_source.shape
    box_xyxy = box_ops.box_cxcywh_to_xyxy(final_box) * torch.Tensor([W, H, W, H])

    cropped_image_np, cropped_mask_np = bbox_crop_image(box_xyxy, image_path, coco, img_id)

    return cropped_image_np, cropped_mask_np


# COCO 데이터셋에서 Bounding Box를 찾는 함수
def COCO_BBOX(image_path, img_id):
    # 각 카테고리에 대한 색상 정의
    color_map = {
        26: 'red',  # joint
        27: 'green',  # gap
    }
    # 각 카테고리에 대한 회색조 값 정의
    grayscale_map = {
        26: 128,  # joint
        27: 255,  # gap
    }

    image = io.imread(image_path)

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    annotations = coco.loadAnns(ann_ids)

    largest_bbox = max(annotations, key=lambda x: x['bbox'][2] * x['bbox'][3])['bbox']

    mask_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # 가장 큰 Bounding Box를 [x, y, width, height] 형식으로 가정
    x, y, w, h = largest_bbox

    # 자르는 사각형이 이미지 경계를 넘지 않도록 조정
    x_end = min(x + w, image.shape[1])
    y_end = min(y + h, image.shape[0])
    x = max(x, 0)
    y = max(y, 0)

    # 계산된 좌표를 사용하여 이미지 자르기
    gt_cropped_image = image[y:y_end, x:x_end]
    gt_cropped_mask = mask_image[y:y_end, x:x_end]

    return gt_cropped_image, gt_cropped_mask
# 이미지를 목표 크기로 조정하는 함수
def resize_image(image, target_size):
    resized_image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return resized_image

# 두 개의 회색조 이미지 사이의 유사성을 계산하는 함수
def calculate_image_similarity(imageA, imageB):
    # 구조적 유사성 지수
    ssim_value = ssim(imageA, imageB)

    # 피크 신호 대 잡음비
    psnr_value = psnr(imageA, imageB)

    metrics = {
        'SSIM': ssim_value,
        'PSNR': psnr_value
    }

    return metrics

# 이미지 전처리 함수
def preprocess_inference(image):
    transform = A.Compose([
        A.Resize(height=1024, width=1024),
        ToTensorV2(),
    ])
    original_size = image.size
    _image = np.array(image.convert("RGB"))
    augmented = transform(image=_image)
    image = augmented['image'].unsqueeze(0)  # 배치 차원 추가
    return image, original_size

# 사전 학습된 모델을 로드하는 함수
def load_pretrained_model(model_path):
    model = deeplabv3_resnet50(pretrained=False, num_classes=3)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'].float().state_dict())

    return model

# 잘린 이미지에 대해 사전 학습된 모델로 추론을 수행하는 함수
def pretrained_inference(cropped_image_np, model):
    model.eval()
    input_image, original_size = preprocess_inference(Image.fromarray(cropped_image_np))
    input_image = input_image.to('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)

    with torch.no_grad():
        output = model(input_image)
        output = output['out']
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()
    if len(output.shape) == 3:
        output = np.transpose(output, (1, 2, 0))
    output_class = np.argmax(output, axis=2)
    output_class_resized = resize(output_class, (original_size[1], original_size[0]), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    return output_class_resized

# 이미지에서 열 기반 샘플링을 수행하는 함수
def column_based_sampling(gimage, mask, num_samples=10, num_columns=10):
    rows, cols = mask.shape
    col_step = cols // num_columns

    sampled_coords = []

    for j in range(num_columns):
        col_start = j * col_step
        col_end = (j + 1) * col_step if (j + 1) * col_step < cols else cols

        region_mask = mask[:, col_start:col_end]
        region_gimage = gimage[:, col_start:col_end]

        region_coords = np.argwhere((region_mask) & (region_gimage != 0))
        if region_coords.size == 0:
            continue

        region_pixel_values = [region_gimage[tuple(coord)] for coord in region_coords]

        region_coords_values = list(zip(region_coords, region_pixel_values))

        best_coord = sorted(region_coords_values, key=lambda x: x[1])[0][0]

        best_coord_adjusted = (best_coord[0], best_coord[1] + col_start)
        sampled_coords.append(best_coord_adjusted)

        if len(sampled_coords) >= num_samples:
            return np.array(sampled_coords)

    return np.array(sampled_coords)

# 이미지에서 너비 기반 샘플링을 수행하는 함수
def width_based_sampling(gimage, mask, num_samples=10, num_rows=10):
    rows, cols = mask.shape
    row_step = rows // num_rows

    sampled_coords = []

    for i in range(num_rows):
        row_start = i * row_step
        row_end = (i + 1) * row_step if (i + 1) * row_step < rows else rows

        region_mask = mask[row_start:row_end, :]
        region_gimage = gimage[row_start:row_end, :]

        region_coords = np.argwhere((region_mask) & (region_gimage != 0))
        if region_coords.size == 0:
            continue

        region_pixel_values = [region_gimage[tuple(coord)] for coord in region_coords]

        region_coords_values = list(zip(region_coords, region_pixel_values))

        best_coord = sorted(region_coords_values, key=lambda x: x[1])[0][0]

        best_coord_adjusted = (best_coord[0] + row_start, best_coord[1])
        sampled_coords.append(best_coord_adjusted)

        if len(sampled_coords) >= num_samples:
            return np.array(sampled_coords)

    return np.array(sampled_coords)

# SAM을 사용하여 이미지를 개선하는 함수
def sam_prompting(output_class_resized, cropped_image_np):
    expansion_joint_mask = (output_class_resized == 1)

    masked_image = np.zeros_like(cropped_image_np)
    masked_image[expansion_joint_mask] = cropped_image_np[expansion_joint_mask]

    window_size = 19
    sauvola_thresh = threshold_sauvola(masked_image, window_size=window_size)
    binary_mask = masked_image < sauvola_thresh
    _mask = morphology.remove_small_objects(binary_mask, 10000)

    column_coordinates = column_based_sampling(cropped_image_np, _mask, num_samples=10, num_columns=10)
    width_coordinates = width_based_sampling(cropped_image_np, _mask, num_samples=10, num_rows=10)

    return np.concatenate((column_coordinates, width_coordinates), axis=0)

# SAM을 사용하여 이미지를 개선하는 함수
def sam_improving(cropped_image_np, coordinates:np.ndarray, sam_predictor):
    image = np.array(Image.fromarray(cropped_image_np).convert("RGB"))

    swapped_coordinates = coordinates[:, [1,0]]
    input_label = np.ones(len(swapped_coordinates)).astype(np.int64)

    sam_predictor.set_image(image)

    masks, scores, logits = sam_predictor.predict(
        point_coords=swapped_coordinates,
        point_labels=input_label,
        multimask_output=True,
    )

    sam_mask = masks[np.argmax(scores)].astype(np.uint8)

    return sam_mask

# 이미지와 마스크의 겹침을 시각화하는 함수
def overlap_visualization(image:np.ndarray, mask:np.ndarray):
    overlay = np.zeros_like(image)
    overlay[mask == 1] = cropped_image_np[mask == 1]
    return overlay

# 평가 메트릭스를 계산하는 함수
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import adapted_rand_error as dice_coefficient
def calculate_metrics(annotation_mask:np.ndarray, inference_mask:np.ndarray):
    annotation_flat = annotation_mask.flatten()
    inference_flat = inference_mask.flatten()

    pixel_accuracy = np.sum(annotation_flat == inference_flat) / len(annotation_flat)

    iou = jaccard_score(annotation_flat, inference_flat)

    dice = dice_coefficient(annotation_flat, inference_flat)[1]

    hausdorff_dist = max(directed_hausdorff(annotation_mask, inference_mask)[0],
                         directed_hausdorff(inference_mask, annotation_mask)[0])

    metrics = {
        'Pixel Accuracy': pixel_accuracy,
        'IoU': iou,
        'Dice Coefficient': dice,
        'Hausdorff Distance': hausdorff_dist
    }

    return metrics

# 메트릭스를 반환하는 함수
def comparison_return_metrics(cropped_mask_np, output_class_resized, sam_mask):
    annotation_mask = (cropped_mask_np == 255).astype(np.uint8)
    trained_mask = (output_class_resized == 1).astype(np.uint8)

    metrics_trained = calculate_metrics(annotation_mask, trained_mask)
    metrics_sam = calculate_metrics(annotation_mask, sam_mask)
    return metrics_trained, metrics_sam




if __name__ == '__main__':
	failed_dino_paths = {}
	# Initialize an empty DataFrame to store the metrics
	metrics_df = pd.DataFrame(
		columns=['image_basename', 'img_id', 'metric_type', 'Pixel Accuracy', 'IoU', 'Dice Coefficient',
				 'Hausdorff Distance'])
	grounding_dino_model, sam_predictor = load_foundation_models()
	pretrained_model = load_pretrained_model('/home/gyumin/mmsegmentation/deeplabv3_resnet50_epoch_125.pth')

	file_name_to_path, coco = image_paths_coco_return(root_dir='./NEXUS_Dataset/NEXUS_3',
													  coco_json_dir='./NEXUS_Dataset/NEXUS_3.json')
	img_ids = coco.getImgIds()
	for img_id in tqdm(img_ids):
		img_info = coco.loadImgs(img_id)[0]
		image_path = file_name_to_path[img_info['file_name']]
		image_basename = os.path.basename(image_path)
		final_box, image_source, image_tensor = DINO_BBOX(image_path, grounding_dino_model)
		if final_box is None:
			print(f"DINO Failed to detect bbox from original image")
			failed_dino_paths[img_id] = image_path
			continue
		cropped_image_np, cropped_mask_np = DINO_CROP(image_source, final_box, coco, img_id)
		gt_cropped_image, gt_cropped_mask = COCO_BBOX(image_path, img_id)

		for_review_resized = resize_image(cropped_image_np, gt_cropped_image.shape)
		metrics = calculate_image_similarity(gt_cropped_image, for_review_resized)
		if metrics['SSIM'] < 0.2:
			output_class_resized = pretrained_inference(cropped_image_np, pretrained_model)
			sampling_coordinates = sam_prompting(output_class_resized, cropped_image_np)
			if sampling_coordinates.size == 0:
				# Check if sampling_coordinates is empty
				print(f"{img_id} / {image_path} failed to find sam_prompt")
				failed_dino_paths[img_id] = image_path
				continue
			sam_mask = sam_improving(cropped_image_np, sampling_coordinates, sam_predictor)
			metrics_trained, metrics_sam = comparison_return_metrics(cropped_mask_np, output_class_resized, sam_mask)

			# Add metrics to DataFrame for trained model
			metrics_trained_df = pd.DataFrame([{
				'image_basename': image_basename,
				'img_id': img_id,
				'metric_type': 'trained',
				'Pixel Accuracy': metrics_trained['Pixel Accuracy'],
				'IoU': metrics_trained['IoU'],
				'Dice Coefficient': metrics_trained['Dice Coefficient'],
				'Hausdorff Distance': metrics_trained['Hausdorff Distance']
			}])
			metrics_df = pd.concat([metrics_df, metrics_trained_df], ignore_index=True)

			# Add metrics to DataFrame for SAM model
			metrics_sam_df = pd.DataFrame([{
				'image_basename': image_basename,
				'img_id': img_id,
				'metric_type': 'SAM',
				'Pixel Accuracy': metrics_sam['Pixel Accuracy'],
				'IoU': metrics_sam['IoU'],
				'Dice Coefficient': metrics_sam['Dice Coefficient'],
				'Hausdorff Distance': metrics_sam['Hausdorff Distance']
			}])
			metrics_df = pd.concat([metrics_df, metrics_sam_df], ignore_index=True)

			print(f"Trained Metrics :\n {metrics_trained} \n SAM Metrics :\n {metrics_sam}")
		else:
			print(f"{img_id} / {image_path} DINO failed to detect box")
			failed_dino_paths[img_id] = image_path

	# Save the DataFrame to a CSV file
	metrics_df.to_csv('metrics_results.csv', index=False)

	# Save the failed_dino_paths to a JSON file
	with open('failed_dino_paths.json', 'w') as f:
		json.dump(failed_dino_paths, f)