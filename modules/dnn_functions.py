from PySide6.QtWidgets import QMainWindow

from .ui_main import Ui_MainWindow
from .utils import cvtPixmapToArray

from mmseg.apis import init_model, inference_model
import numpy as np
import skimage.morphology

from segment_anything import sam_model_registry, SamPredictor

from submodules.SAM2.sam2.build_sam import build_sam2
from submodules.SAM2.sam2.sam2_image_predictor import SAM2ImagePredictor

from submodules.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

import torch

class DNNFunctions(object):
    def __init__(self):

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        ######################### 
        # Semantic Segmentation #
        #########################
        
        # MMSegmentation
        self.mmseg_config = 'dnn/configs/promptModel.py'
        self.mmseg_checkpoint = 'dnn/checkpoints/promptModel.pth'
        
        # MMSegmentation
        self.mmseg_config_v2 = 'dnn/configs/promptModelV2.py'
        self.mmseg_checkpoint_v2 = 'dnn/checkpoints/promptModelV2.pth'

        # Segment Anything
        self.sam_checkpoint = 'dnn/checkpoints/sam_vit_h_4b8939.pth'

        self.sam2_checkpoint = 'dnn/checkpoints/sam2.1_hiera_large.pt'
        self.sam2_config = 'C:/Users/tls15/Documents/01.Program/JGAMS_2/dnn/configs/sam2.1_hiera_l.yaml'

        #################### 
        # Object Detection #
        ####################
        
        # GroundingDINO
        self.groundingDino_config = 'dnn/configs/GroundingDINO_SwinB_cfg.py'
        self.groundingDino_checkpoint = 'dnn/checkpoints/groundingdino_swinb_cogcoor.pth'

        ##############
        # Attributes #
        ##############
        
        self.scale = 1.0

    def load_groundingDino(self, config, checkpoint):
        self.groundingDino_model = load_model(config, checkpoint)
    
    def inference_groundingDino(self, model, image, caption, box_threshold, text_threshold, device="cuda"):

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
            )
        
        return boxes, logits, phrases


    def load_sam(self, checkpoint, mode='default'):
        """
        Load the sam model
        Args:
            mode (str): The mode of the sam model.
        """
        self.sam_model = sam_model_registry[mode](checkpoint=checkpoint)
        self.sam_model.to(device='cuda:0')
        self.sam_predictor = SamPredictor(self.sam_model)
    
    def load_sam2(self, config, checkpoint, mode='default'):
        """
        Load the sam model
        Args:
            mode (str): The mode of the sam model.
        """

        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.sam2_model = build_sam2(config, checkpoint, device='cuda:0')
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def set_sam_image(self):
        image = cvtPixmapToArray(self.pixmap)
        image = image[:, :, :3]
        
        self.sam_predictor.set_image(image)


    def load_mmseg(self, config_file, checkpoint_file):
        """
        Load the mmseg model
        Args:
            config_file (str): The path to the config file.
            checkpoint_file (str): The path to the checkpoint file.
        """
        self.mmseg_model = init_model(config_file, checkpoint_file, device='cuda:0')
        
    def inference_mmseg(self, img, do_crf=True):
        """
        Inference the image with the mmseg model

        Args:
            img (np.ndarray): The image to be processed.
            do_crf (bool): Whether to apply DenseCRF.

        Returns:
            mask (np.ndarray): The processed mask.

        """
        # filter image size too small or too large
        # if img.shape[0] < 50 or img.shape[1] < 50 :
        #     print(f"too small")
        #     return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # elif img.shape[0] > 1000 or img.shape[1] > 1000 :
        #     print(f"too large")
        #     return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        img = self.cvtRGBATORGB(img)

        result = inference_model(self.mmseg_model, img)

        mask = result.pred_sem_seg.data.cpu().numpy()
        mask = np.squeeze(mask)

        logits = result.seg_logits.data.cpu().numpy()
        
        back = mask == 0
        joint = mask == 1
        gap = mask == 2

        back = skimage.morphology.binary_closing(back, skimage.morphology.square(3))
        joint = skimage.morphology.binary_closing(joint, skimage.morphology.square(3))
        gap = skimage.morphology.binary_closing(gap, skimage.morphology.square(3))
        
        return back, joint, gap, logits
    
    @staticmethod
    def cvtRGBATORGB(img):
        """Convert a RGBA image to a RGB image
        Args:
            img (np.ndarray): The image to be converted.

        Returns:
            img (np.ndarray): The converted image.
        
        """
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    

    


    