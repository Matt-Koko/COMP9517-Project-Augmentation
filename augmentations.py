"""
Contains all the augmentation transformations for each model.

Also contains the code for the unique CutNPaint method
"""

import numpy as np
import albumentations as A
import torch
import torchvision.transforms.functional as tvF
from torchvision import transforms
from PIL import Image
import random
import torchvision.utils as vutils
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn

from Inpainter.model.networks import Generator
from Inpainter.utils.tools import normalize

"""
Small resnet transformations
"""

smallresnet_pre_resize = A.Compose([
        A.HorizontalFlip(),
        A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.05, hue=0.05, always_apply=True),
        A.FancyPCA(always_apply=True)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

smallresnet_post_resize = A.Compose([
        A.HorizontalFlip(),
        A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.05, hue=0.05, always_apply=True),
        A.FancyPCA(always_apply=True)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

augmented_ds_1 = A.Compose([
        A.HorizontalFlip(),
        A.RandomSunFlare(p=0.05),
        A.RandomRain(p=0.05),
        A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.05, hue=0.05, always_apply=True),
        A.FancyPCA(always_apply=True),
        A.GaussNoise(p=0.1)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

augmented_ds_2 = A.Compose([
        A.BBoxSafeRandomCrop(p=0.5),
        A.Perspective(p=0.2),
        A.HorizontalFlip(),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, value=[255, 255, 255], always_apply=True)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

"""
Cut n Paint Custom Augmentation
"""

# Inpainting variables
netG_config =  {'input_dim': 3, 'ngf': 32}
cuda_device_ids = [0]
model_name = './inpainter/checkpoints/imagenet/hole_benchmark/gen_00430000.pt'
netG = Generator(netG_config, torch.cuda.is_available(), cuda_device_ids)   
netG.load_state_dict(torch.load(model_name))
if torch.cuda.is_available():
    netG = nn.parallel.DataParallel(netG, device_ids=cuda_device_ids)


def CutNPaint(image, bbox, label, preview_mode=False):
    # Images from datasets are read only
    image = np.copy(image)
    
    # Show the original image
    if preview_mode:
        #show(image, bbox, "Original")
        fig, axs = plt.subplots(1, 3)
        axs[0].set_title("Original")
        axs[0].imshow(image)
        bboxPatch = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        axs[0].add_patch(bboxPatch)

    image_height, image_width, _ = image.shape

    # Augmentation operation to perform on the cropped penguin
    bbox = [round(x) for x in bbox]
    zoom_augment = A.Compose([
        A.Crop(x_min=bbox[0], y_min=bbox[1], x_max=bbox[0] + bbox[2], y_max=bbox[1] + bbox[3]),
        A.HorizontalFlip(),
        A.RandomScale(scale_limit=[-0.9, 0], always_apply=True)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    # Perform the above transformation to get the penguin
    subject_augmented_data = zoom_augment(image=image, bboxes=[bbox], category_ids=[label]) # Albumentations
    subject_image = subject_augmented_data['image']
    subject_bbox = list(subject_augmented_data['bboxes'][0])
    # Show the removed images
    if preview_mode:
        #image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :] = 0   # Just for show
        #show(image, None, "Subject Masked")
        
        axs[1].set_title("Subject Masked")
        image2 = np.copy(image)
        image2[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :] = 0
        axs[1].imshow(image2)

    # Perform geometric transformations on the cut out penguin or turtle and paste it in
    paste_bbox = np.zeros(4, dtype=int)
    paste_bbox[0] = random.randint(0, image_width - bbox[2])  # Can move anywhere on the x axis
    paste_bbox[3], paste_bbox[2], _ = subject_image.shape

    # Only perturb the height a little bit
    v_padding = round(0.2 * image_height)
    paste_y_min = v_padding
    paste_y_max = image_height - v_padding - paste_bbox[3]
    if paste_y_min < paste_y_max:
        paste_bbox[1] = random.randint(paste_y_min, paste_y_max)

    # Paste the cropped image in at a new random horizontal location
    #image[paste_bbox[1]:paste_bbox[1] + paste_bbox[3], paste_bbox[0]:paste_bbox[0] + paste_bbox[2], :] = subject_image

    # Create a 2D mask of the space requiring infill
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Specify the part of the mask for inpainting
    mask_padding = 5
    x_min = max(0, bbox[0] - mask_padding)
    y_min = max(0, bbox[1] - mask_padding)
    x_max = min(image_width, bbox[0] + bbox[2] + mask_padding)
    y_max = min(image_height, bbox[1] + bbox[3] + mask_padding)
    mask[y_min:y_max, x_min:x_max, :] = 255  # Set the inpainting mask to fill in where the subject was

    #mask[paste_bbox[1]:paste_bbox[1] + paste_bbox[3], paste_bbox[0]:paste_bbox[0] + paste_bbox[2], :] = 0

    
    inpainted_image = inpaint(image, mask)
    inpainted_image[paste_bbox[1]:paste_bbox[1] + paste_bbox[3], paste_bbox[0]:paste_bbox[0] + paste_bbox[2], :] = subject_image
    # Show the removed images
    if preview_mode:
        #show(inpainted_image, paste_bbox, "Augmented Image")
        axs[2].set_title("Inpainted Image with Modified Subject")
        axs[2].imshow(inpainted_image)
        bboxPatch = patches.Rectangle((paste_bbox[0], paste_bbox[1]), paste_bbox[2], paste_bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        axs[2].add_patch(bboxPatch)
        plt.tight_layout()
        plt.show()

    return inpainted_image, paste_bbox


def inpaint(image, mask):
    """
    Give credit here
    """

    # May have to get a window around the image
    image_shape = list(image.shape)
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    x = image
    x = transforms.Resize(image_shape[:-1])(x)
    x = transforms.CenterCrop(image_shape[:-1])(x)
    mask = transforms.Resize(image_shape[:-1])(mask)
    mask = transforms.CenterCrop(image_shape[:-1])(mask)
    x = transforms.ToTensor()(x)
    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
    x = normalize(x)
    x = x * (1. - mask)
    x = x.unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0)

    if torch.cuda.is_available():
        x = x.cuda()
        mask = mask.cuda()

    # Inference
    x1, x2, offset_flow = netG(x, mask)
    inpainted_result_tensor_array = x2 * mask + x * (1. - mask)

    inpainted_result_tensor = vutils.make_grid(inpainted_result_tensor_array, padding=0, normalize=True)
    inpainted_result =  np.array(tvF.to_pil_image(inpainted_result_tensor))
    
    return inpainted_result


def show(image, bbox, title):
    plt.gca().set_title(title)
    plt.imshow(image)
    if bbox is not None:
        bboxPatch = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(bboxPatch)
    plt.show()