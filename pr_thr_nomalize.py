import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model, show_result_pyplot

import torch
import copy
import cv2
from .modules.utils import *

# Set up the argument parser
parser = argparse.ArgumentParser(description="Process image segmentation.")
parser.add_argument('config_path', type=str, help='Path to the config file.')
parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file.')
parser.add_argument('--srx_dir', type=str, help='Directory containing input images.')
parser.add_argument('--rst_dir', type=str, help='Directory to save output images and data.')
parser.add_argument('--threshold', type=float, default=0.50)
args = parser.parse_args()

# Initialize the segmentor using provided config and checkpoint
segmentor = init_model(args.config_path, args.checkpoint_path, device='cuda')



# Loop through all images in the source directory
for image_name in os.listdir(args.srx_dir):
    if image_name.lower().endswith(('.png', '.jpg')):
        image_path = os.path.join(args.srx_dir, image_name)

        temp_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        img = imread(image_path, checkImg=True)
        gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

        palette = np.array([[0, 0, 0],
               [0, 255, 0],
               [0, 0, 255]
               ])


        # Perform inference on the image
        result = inference_model(segmentor, image_path)

        print(f"result: {result}")

        thr = args.threshold

        class_1_logits = result.seg_logits.data.cpu().numpy()[1, :, :]  # Access the second class logits
        
        class_2_logits = result.seg_logits.data.cpu().numpy()[2, :, :]  # Access the second class logits
        
        #logits to prob
        class_1_prob = logits_np_to_prob(class_1_logits)

        class_2_prob = logits_np_to_prob(class_2_logits)

        #logits to min max nomalize score
        class_1_score = min_max_normalize(class_1_logits)

        class_2_score = min_max_normalize(class_2_logits)


        print(f"class_1_lg: {class_1_logits}, class_2_lg: {class_2_logits}")

        print(f"class_1_pb: {class_1_prob}, class_2_pb: {class_2_prob}")

        print(f"class_1_score: {class_1_score}, class_2_score: {class_2_score}")

        


        # class_1_bi = extract_values_above_threshold(class_1_prob_minmax, thr)
        # class_2_bi = extract_values_above_threshold(class_2_prob_minmax, thr)

        class_1_bi = extract_values_above_threshold(class_1_score, thr)
        class_2_bi = extract_values_above_threshold(class_2_score, thr)



        print(f"class_1_bi: {class_1_bi}, class_2_bi: {class_2_bi}")

        idx_1 = np.argwhere(class_1_bi == True)
        idx_2 = np.argwhere(class_2_bi == True)

        y_idx_1, x_idx_1 = idx_1[:, 0], idx_1[:, 1]
        y_idx_2, x_idx_2 = idx_2[:, 0], idx_2[:, 1]
        print(f"idx 1: {idx_1}, idx 2: {idx_2}")
        
        gt[y_idx_1, x_idx_1] = 1
        gt[y_idx_2, x_idx_2] = 2

        
        color_map = blendImageWithColorMap(img, gt, palette, 0.5)

        save_dir = os.path.join(args.rst_dir, f'{os.path.splitext(image_name)[0]}_thr_{thr}.png')

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        
        imwrite(save_dir, color_map)
        

        