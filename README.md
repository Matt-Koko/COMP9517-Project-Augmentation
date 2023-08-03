# Augmentation Guide

### Attributions
The 'Cut N Paint' method is based directly on a [Pytorch implementation](https://github.com/daa233/generative-inpainting-pytorch) by 'daa233' of a paper named [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892).



## Augmented Datasets
All static augmented datasets are contained in the folder "Augmented Datasets" in this directory. They have been replicated with the same file structure as the original dataset and have corresponding annotations. All transformations attempt to retain the context of a photograph.
- **augmented_ds_1**: Non geometric transformations (except horizontal flip).
    - Random horizontal flip
    - Random sun flare (p=0.05)
    - Random rain (p=0.05)
    - Color jittering with emphasis on brightness.
    - FancyPCA from Krizhevsky's paper "ImageNet Classification with Deep Convolutional Neural Networks"
    - GaussNoise (p=0.10)
- **augmented_ds_2**: Geometric transformations.
    - Random bbox safe crop (p=0.4)
    - Random perspective transform (p=0.2)
    - Random horizontal flip
    - Pad to return to original size
- **augmented_ds_3**: 'Cut N Paint' novel method alone.


