# Augmentation Module

## augment.py

This module is used to apply rotation augmentations to an image dataset. It recursively searches a directory for images and saves the augmented versions of those images in a separate target directory—preserving directory structure.

### Rotations
The rotations applied are 0, 90, 180, 270 (+- 5) degress, resulting in 12 varients of each image.

### Usage
1. Move you labeled files to `cap_labeled`
2. Run this
```
python3 augment.py
```
3. You have your augmented images in `cap_augmented`


### Advanced Usage
```
python3 augment.py -h
```
