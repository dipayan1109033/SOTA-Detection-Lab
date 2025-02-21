import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import albumentations as A

from src.utils.common_utils import Helper
helper = Helper()



# A.SmallestMaxSize(max_size=640, p=1.0)
# A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=0)
# A.RandomScale(scale_limit=0.1, p=1.0)
# A.Rotate(limit=30, p=1.0)

def get_augmentation_pipeline():

    augment_pipeline = A.Compose([
            A.OneOf([
                        A.RandomBrightnessContrast(p=0.5),
                        A.HueSaturationValue(p=0.25),
                    ], p=0.25),
            A.OneOf([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.25),
                    ], p=0.5),
            A.OneOf([
                        A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR, p=0.25),
                        A.ShiftScaleRotate(rotate_limit=15, p=0.5),
                        A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.5),
                    ], p=0.5),
            A.XYMasking(num_masks_x=(1,2), num_masks_y=(1,2), mask_x_length=(10, 50), mask_y_length=(10,50), p=0.15)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=100, min_visibility=0.1, label_fields=['labels']),
        p=0.90)

    return augment_pipeline