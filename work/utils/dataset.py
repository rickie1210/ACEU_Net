from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import paddle
from paddle.io import Dataset
import logging
from PIL import Image
from PIL import ImageEnhance
import random

from paddle.vision.transforms import functional as TF

class JointTransform:
    """A simple class to jointly transform images and masks with random flips and rotation."""
    def __init__(self, horizontal_flip_prob=0.1, vertical_flip_prob=0.1):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, image, mask):
        # Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() < self.vertical_flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        image = ImageEnhance.Contrast(image).enhance(1.1)
        # Fixed brightness enhancement
        image = ImageEnhance.Brightness(image).enhance(1.3)
        return image, mask


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform or JointTransform()
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.mds = [splitext(file)[0] for file in listdir(masks_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info(f'masks_dir {len(self.ids)} examples')

        super().__init__()  # 确保也调用了paddle.io.Dataset的初始化

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        # 数据增强
        if self.transform:
            img, mask = self.apply_transforms(img, mask)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': paddle.to_tensor(img, dtype='float32'),
            'mask': paddle.to_tensor(mask, dtype='float32')
        }

    def apply_transforms(self, img, mask):
        img, mask = self.transform(img, mask)
        return img, mask


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
