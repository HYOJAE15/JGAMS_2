import cv2
import numpy as np
import matplotlib.pyplot as plt

from .modules.utils import *
from .modules.utils_img import (
    annotate_GD, getScaledPoint, getScaledPoint_mmdet, getCoordBTWTwoPoints, applyBrushSize, readImageToPixmap)

from modules.utils import imwrite_colormap

from submodules.GroundingDINO.groundingdino.util import box_ops

import torch

import skimage.measure
import skimage.filters
from skimage import morphology

import copy


import argparse
import os.path as osp

import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

from mmseg.apis import init_model, inference_model

from submodules.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate


def parse_args():
    parser = argparse.ArgumentParser(
        description='inference with cityscapes format code'
    )
    # parser.add_argument('config', help='config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'img_dir',
        help=('image directory for inference'))
    parser.add_argument(
        'save_dir',
        help=('directory to save result'))
    # parser.add_argument(
    #     'dataset_type',
    #     help=('data type to be inference'))
    parser.add_argument(
        'img_format',
        help=('format of img'))


    args = parser.parse_args()

    return args

def inferenceGroundingDino(self):

    groundingDino_config = 'dnn/configs/GroundingDINO_SwinB_cfg.py'
    groundingDino_checkpoint = 'dnn/checkpoints/groundingdino_swinb_cogcoor.pth'

    groundingDino_model = load_model(groundingDino_config, groundingDino_checkpoint)

    GD_img_source, GD_img = imread_GD(self.imgPath)
    
    TEXT_PROMPT = "Steel joint" 
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.20


    boxes, logits, phrases = self.inference_groundingDino(model=self.groundingDino_model, 
                                                            image=GD_img,
                                                            caption=TEXT_PROMPT,
                                                            box_threshold=BOX_TRESHOLD,
                                                            text_threshold=TEXT_TRESHOLD,
                                                            )
    
    annotated_frame = annotate_GD(image_source=GD_img_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    index = logits.argmax()
    box = boxes[index]

    # box : normalized box xywh -> unnormalized xyxy
    H, W, _ = GD_img_source.shape
    box_xyxy = box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H])

    # crop image
    print(box_xyxy)
    min_x, min_y, max_x, max_y = box_xyxy.int().tolist()
    print(f"{min_x}, {min_y}, {max_x}, {max_y}")

    img = cvtPixmapToArray(self.pixmap)
    img_roi = img[min_y:max_y, min_x:max_x, :3]
    
    # _colormap = copy.deepcopy(self.colormap)        
    # _colormap = cv2.rectangle(_colormap, (min_x, min_y), (max_x, max_y), (255, 255, 255, 255), 15)

    # self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
    # self.color_pixmap_item.setPixmap(QPixmap())
    # self.color_pixmap_item.setPixmap(self.color_pixmap)

    self.GD_min_x = min_x
    self.GD_min_y = min_y
    self.GD_max_x = max_x
    self.GD_max_y = max_y

    self.promptModel()
    
    ## 2. Create SAM's Prompt 
def promptModel(self):
    self.load_mmseg(self.mmseg_config, self.mmseg_checkpoint)
    
    img = cvtPixmapToArray(self.pixmap)
    self.GD_img_roi = img[self.GD_min_y:self.GD_max_y, self.GD_min_x:self.GD_max_x, :3]
    
    back, joint, gap, logits = self.inference_mmseg(self.GD_img_roi)


    
    """
    nomalize the segmentation logits
    """
    # background
    # back_logit = logits[0, :, :]
    # back_score = min_max_normalize(back_logit)
    # back_bi = extract_values_above_threshold(back_score, thr)
    
    # back_idx = np.argwhere(back_bi == 1)
    # back_y_idx, back_x_idx = back_idx[:, 0], back_idx[:, 1]
    # back_x_idx = back_x_idx + self.GD_min_x
    # back_y_idx = back_y_idx + self.GD_min_y

    # self.label[back_y_idx, back_x_idx] = 0
    # self.colormap[back_y_idx, back_x_idx, :3] = self.label_palette[0]

    # joint
    joint_logit = logits[1, :, :]
    joint_score = min_max_normalize(joint_logit)
    joint_bi = extract_values_above_threshold(joint_score, self.pred_thr)
    joint_bi = morphology.remove_small_objects(joint_bi, self.area_thr)
    joint_bi = morphology.remove_small_holes(joint_bi, self.fill_thr)
    
    joint_idx = np.argwhere(joint_bi == 1)
    joint_y_idx, joint_x_idx = joint_idx[:, 0], joint_idx[:, 1]
    joint_x_idx = joint_x_idx + self.GD_min_x
    joint_y_idx = joint_y_idx + self.GD_min_y

    self.label[joint_y_idx, joint_x_idx] = 1
    self.colormap[joint_y_idx, joint_x_idx, :3] = self.label_palette[1]

    # gap
    gap_logit = logits[2, :, :]
    gap_score = min_max_normalize(gap_logit)
    gap_bi = extract_values_above_threshold(gap_score, self.pred_thr)
    gap_bi = morphology.remove_small_objects(gap_bi, self.area_thr)
    gap_bi = morphology.remove_small_holes(gap_bi, self.fill_thr)
    
    gap_idx = np.argwhere(gap_bi == 1)
    gap_y_idx, gap_x_idx = gap_idx[:, 0], gap_idx[:, 1]
    gap_x_idx = gap_x_idx + self.GD_min_x
    gap_y_idx = gap_y_idx + self.GD_min_y

    self.label[gap_y_idx, gap_x_idx] = 2
    self.colormap[gap_y_idx, gap_x_idx, :3] = self.label_palette[2]

    img = img[:, :, :3]
    prompt_colormap = blendImageWithColorMap(img, self.label) 

    promptPath = self.imgPath.replace('/leftImg8bit/', '/promptLabelIds/')
    promptPath = promptPath.replace( '_leftImg8bit.png', f'_prompt({self.pred_thr})_labelIds.png')
    promptColormapPath = promptPath.replace(f'_prompt({self.pred_thr})_labelIds.png', f"_prompt({self.pred_thr})_color.png")        
    os.makedirs(os.path.dirname(promptPath), exist_ok=True)
    
    print(f"prompt result: {promptPath}, {promptColormapPath}")
    imwrite(promptPath, self.label) 
    imwrite_colormap(promptColormapPath, prompt_colormap)

    

        
    # self.colormap[gap_y_idx, gap_x_idx, :3] = self.label_palette[2]

    
    # _colormap = copy.deepcopy(self.colormap)
    # cv2.rectangle(_colormap, (self.GD_min_x, self.GD_min_y), (self.GD_max_x, self.GD_max_y), (255, 255, 255, 255), 3)

    # self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
    # self.color_pixmap_item.setPixmap(QPixmap())
    # self.color_pixmap_item.setPixmap(self.color_pixmap)

    self.pointSampling()

    ## 2.1. Point Sampling
def pointSampling(self):
    
    img = np.array(Image.open(self.imgPath))

    joint = self.label == 1
    # joint = morphology.erosion(joint, morphology.square(15))
    gap = self.label == 2
    # gap = morphology.erosion(gap, morphology.square(15))
    

    joint_col_coord = column_based_sampling(img, joint, num_samples=10, num_columns=10)
    joint_row_coord = width_based_sampling(img, joint, num_samples=10, num_rows=10)

    joint_coord = np.concatenate((joint_col_coord, joint_row_coord), axis=0)
    joint_coord = joint_coord[:, [1,0]]
    joint_label = np.zeros((joint_coord.shape[0]), dtype=int)

    gap_col_coord = column_based_sampling(img, gap, num_samples=10, num_columns=10)
    gap_row_coord = width_based_sampling(img, gap, num_samples=10, num_rows=10)

    gap_coord = np.concatenate((gap_col_coord, gap_row_coord), axis=0)
    gap_coord = gap_coord[:, [1,0]]
    gap_label = np.ones((gap_coord.shape[0]), dtype=int)

    input_point = np.concatenate((gap_coord, joint_coord), axis=0)
    input_label = np.concatenate((gap_label, joint_label), axis=0)
    
    input_box = np.array([self.GD_min_x, self.GD_min_y, self.GD_max_x, self.GD_max_y])

    
    if hasattr(self, 'sam_model') == False :
        self.load_sam(self.sam_checkpoint) 

    img = cvtPixmapToArray(self.pixmap)
    img = img[:, :, :3]
            
    self.sam_predictor.set_image(img)
    
    
    masks, scores, logits = self.sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box, 
        multimask_output=True,
    )

    mask = masks[np.argmax(scores), :, :]
    self.sam_mask_input = logits[np.argmax(scores), :, :]

    # update label with result
    idx = np.argwhere(mask == 1)
    y_idx, x_idx = idx[:, 0], idx[:, 1]

    self.GD_sam_y_idx = y_idx
    self.GD_sam_x_idx = x_idx
    
    self.label[self.label==1] = 0
    
    self.updateColorMap()

    self.label[y_idx, x_idx] = 2
    self.colormap[y_idx, x_idx, :3] = self.label_palette[2]

    _colormap = copy.deepcopy(self.colormap)

    for joint in joint_coord:
        
        # cv2.circle(_colormap, (joint[0], joint[1]), 50, (0, 0, 255, 255), 9)
        cv2.circle(_colormap, (joint[0], joint[1]), 9, (0, 0, 255, 255), -1)
    for gap in gap_coord:
        
        # cv2.circle(_colormap, (gap[0], gap[1]), 50, (255, 0, 0, 255), 9)
        cv2.circle(_colormap, (gap[0], gap[1]), 9, (255, 0, 0, 255), -1)
    


    self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
    self.color_pixmap_item.setPixmap(QPixmap())
    self.color_pixmap_item.setPixmap(self.color_pixmap)

    sam_colormap = blendImageWithColorMap(img, self.label) 
    img = imread(self.imgPath)

    for joint in joint_coord:
        
        cv2.circle(sam_colormap, (joint[0], joint[1]), 9, (255, 0, 0, 255), -1)
        cv2.circle(img, (joint[0], joint[1]), 9, (255, 0, 0, 255), -1)
    for gap in gap_coord:
        
        cv2.circle(sam_colormap, (gap[0], gap[1]), 9, (0, 0, 255, 255), -1)
        cv2.circle(img, (gap[0], gap[1]), 9, (0, 0, 255, 255), -1)
    
    
    colormapPath = os.path.dirname(self.labelPath)
    colormapName = os.path.basename(self.labelPath)
    colormapPath = os.path.dirname(colormapPath)
    colormapPath = os.path.dirname(colormapPath)
    colormapPath = os.path.join(colormapPath, "JGAM_colormap")
    os.makedirs(colormapPath, exist_ok=True)
    colormapPath = os.path.join(colormapPath, colormapName)

    pointName = colormapName.replace("_labelIds.png", "point.png")
    pointmapPath = os.path.join(os.path.dirname(colormapPath), pointName)


    imwrite_colormap(colormapPath, sam_colormap)
    cv2.imwrite(pointmapPath, img)
    
    print(f"colormapPath: {colormapPath}, pointmapPath:{pointmapPath}")
    



def main():
    args = parse_args()
    
    # model = init_model(args.config, args.checkpoint, device='cuda:0')

    img_list = glob(osp.join(args.img_dir, f'*.{args.img_format}'))

    print(img_list[0])

    save_dir = args.save_dir



if __name__ == '__main__':
    main()
