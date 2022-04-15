import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import random

from settings import keypoints_classes_ids2names

def train_transform():
    return A.Compose([
        # A.Sequential([
        #     A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
        #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        # ], p=1)
        A.Sequential([
            A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(scale_limit=0, rotate_limit=5, shift_limit=0.5, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            A.HueSaturationValue(p=0.3,hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=20),
            A.CLAHE(p=0.3, clip_limit=4.0, tile_grid_size=(8, 8)),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1),
            A.Blur(p=0.3, blur_limit=3),
            # A.PadIfNeeded(min_height=416, min_width=416, always_apply=True), # @TODO
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        f = plt.figure(figsize=(40,40))
        plt.imshow(image)
        plt.close()
        return f

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

        plt.close()

        return f