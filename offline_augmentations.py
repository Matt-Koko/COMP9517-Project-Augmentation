"""
Generate augmented training datasets for use offline
"""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvF

from augmentations import augmented_ds_1, augmented_ds_2, CutNPaint, show

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AugmentationDataset(Dataset):
    def __init__(self, data_frame, original_train_path):
        self.original_path = original_train_path
        self.image_names = data_frame['image_filenames']
        self.id = data_frame['id']
        self.labels = data_frame['category_id'].values
        self.x_min = data_frame['x']
        self.y_min = data_frame['y']
        self.width = data_frame['w']
        self.height = data_frame['h']
        self.area = data_frame['area']
        self.augmentation_previews = 0

    def __len__(self):
        return len(self.image_names)

    def get_orig_size(self,idx):
        return Image.open(self.image_paths[idx]).size
    
    def get_bbox(self, idx):
        return [self.x_min[idx], self.y_min[idx], self.width[idx], self.height[idx]]
        
    def __getitem__(self, idx): 
        image = Image.open(os.path.join(self.original_path, self.image_names[idx]))
        image = image.convert('RGB')
        image = np.asarray(image)

        label = self.labels[idx]
        bbox = self.get_bbox(idx)

        return image, bbox, label
    
    def preview_generative_inpainting(self, num_previews):
        '''
        Randomly show examples of CutNPaint
        '''
        for i in range(num_previews):
            idx = random.randint(0, len(self) - 1)

            image, bbox, label = self[idx]
            CutNPaint(image, bbox, label, preview_mode=True)

    def export_static_dataset(self, parent_directory, augmentation_transform):

        # Set the export paths
        annotation_path = os.path.join(parent_directory, 'train_annotations')
        image_save_path = os.path.join(parent_directory, 'train\\train\\')

        annotation_data = []
        # Loop through each image, 
        for idx, data in enumerate(self):
            image, bbox, label = data

            # apply the augmentation,
            augmented_image_data = augmentation_transform(image=image, bboxes=[bbox], category_ids=[label])
            image = augmented_image_data['image']        
            bbox_new = list(augmented_image_data['bboxes'][0])


            # append new annotations to list
            bbox_aug = [int(item) for item in bbox_new]
            bbox_aug_area = int(bbox_aug[2] * bbox_aug[3])
            id = int(self.id[idx])
            prediction_dict = {
                "id": id,
                "image_id": id,   # Image ID in the annotations is always the same as id
                "category_id": int(label) + 1, # Normalized back to 0, 1
                "bbox": bbox_aug,
                "area": bbox_aug_area,
                "segmentation": [],
                "iscrowd": 0 
            }
            annotation_data.append(prediction_dict)

            # save in the augmentation folder
            if self.augmentation_previews > 0:
                self.augmentation_previews -= 1
                show(image, bbox_new, "Test")
            
            # Convert back to Image and save
            PIL_image = Image.fromarray(image)
            PIL_image.save(os.path.join(image_save_path, self.image_names[idx]))

            if idx == len(self) - 1:
                break

        # Export the annotations
        with open(annotation_path, "w") as file:
            json.dump(annotation_data, file)
        print("Annotations exported to", annotation_path)


# Setup the training dataset
original_train_path = "../Datasets/original_data/train/train"
train_annotation = '../Datasets/original_data/train_annotations'

train = pd.read_json(train_annotation)
train['image_filenames'] = os.listdir(original_train_path)
train[['x','y','w','h']] = train['bbox'].apply(pd.Series)

# Generate a static offline dataset
ds = AugmentationDataset(train, original_train_path)
#print(len(ds))
ds.export_static_dataset(".\\Augmented Datasets\\augmented_ds_2\\", augmented_ds_2)

def preview_generative_inpainting(num_previews):
    """
    For calls from `cut_n_paint_preview.py' (used when following README instructions)
    """
    ds.preview_generative_inpainting(num_previews)