import random

import numpy as np
import torch
from torch import Tensor


def _fix_negative_strides(img):
    return np.ascontiguousarray(img)


class Rotate:
    """
    Randomly rotates a single image or a set of N images by multiples of 90 degrees (same augmentation for all images
    in the set).
    """

    def __init__(self):
        # Number of 90 degrees rotations counter-clockwise
        self.k = random.randint(0, 3)

    def __call__(self, img: Tensor) -> Tensor:
        """
        Args:
            img:   torch tensor, accepted dimensions: [N, C, H, W], [C, H, W], or [H, W],
                   where:
                      N: number of images
                      C: number of channels
                      H: image height
                      W: image width

        Returns:
            img_rotated:    torch tensor of the same dimensions, where the images (last two dimensions) are randomly
                            rotated by multiples of 90 degrees.
        """

        img_rotated = torch.zeros_like(img)

        if img.dim() == 4:
            for sample in range(img.shape[0]):
                for channel in range(img.shape[1]):
                    rotated = np.rot90(img[sample, channel, ...].numpy(), self.k)
                    img_rotated[sample, channel, ...] = torch.from_numpy(_fix_negative_strides(rotated))

        elif img.dim() == 3:
            for channel in range(img.shape[0]):
                rotated = np.rot90(img[channel, ...].numpy(), self.k)
                img_rotated[channel, ...] = torch.from_numpy(_fix_negative_strides(rotated))
        else:
            rotated = np.rot90(img.numpy(), self.k)
            img_rotated = torch.from_numpy(_fix_negative_strides(rotated))

        return img_rotated

class RandomHorizontalFlip:
    """
    Randomly flips an image or a set of N images along the horizontal axis (same augmentation for all images in the set).
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img: Tensor) -> Tensor:
        """
        Args:
            img:   torch tensor, accepted dimensions: [N, C, H, W], [C, H, W], or [H, W],
                   where:
                      N: number of images
                      C: number of channels
                      H: image height
                      W: image width

        Returns:
            img_flipped:    torch tensor of the same dimensions, where the images (last two dimensions) are randomly
                            horizontally flipped.
        """

        img_flipped = torch.zeros_like(img)

        do_flip = random.random() < self.prob

        if img.dim() == 4:
            for sample in range(img.shape[0]):
                for channel in range(img.shape[1]):
                    if do_flip:
                        flipped = np.fliplr(img[sample, channel, ...].numpy())
                        img_flipped[sample, channel, ...] = torch.from_numpy(_fix_negative_strides(flipped))
                    else:
                        img_flipped[sample, channel, ...] = img[sample, channel, ...].clone()

        elif img.dim() == 3:
            for channel in range(img.shape[0]):
                if do_flip:
                    flipped = np.fliplr(img[channel, ...].numpy())
                    img_flipped[channel, ...] = torch.from_numpy(_fix_negative_strides(flipped))
                else:
                    img_flipped[channel, ...] = img[channel, ...].clone()

        else:
            if do_flip:
                flipped = np.fliplr(img.numpy())
                img_flipped = torch.from_numpy(_fix_negative_strides(flipped))
            else:
                img_flipped = img.clone()

        return img_flipped


class RandomVerticalFlip:
    """
    Randomly flips an image or a set of N images along the vertical axis (same augmentation for all images in the set).
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img: Tensor) -> Tensor:
        """
        Args:
            img:   torch tensor, accepted dimensions: [N, C, H, W], [C, H, W], or [H, W],
                   where:
                      N: number of images
                      C: number of channels
                      H: image height
                      W: image width

        Returns:
            img_flipped:    torch tensor of the same dimensions, where the images (last two dimensions) are randomly
                            vertically flipped.
        """

        img_flipped = torch.zeros_like(img)

        do_flip = random.random() < self.prob

        if img.dim() == 4:
            for sample in range(img.shape[0]):
                for channel in range(img.shape[1]):
                    if do_flip:
                        flipped = np.flipud(img[sample, channel, ...].numpy())
                        img_flipped[sample, channel, ...] = torch.from_numpy(_fix_negative_strides(flipped))
                    else:
                        img_flipped[sample, channel, ...] = img[sample, channel, ...].clone()

        elif img.dim() == 3:
            for channel in range(img.shape[0]):
                if do_flip:
                    flipped = np.flipud(img[channel, ...].numpy())
                    img_flipped[channel, ...] = torch.from_numpy(_fix_negative_strides(flipped))
                else:
                    img_flipped[channel, ...] = img[channel, ...].clone()

        else:
            if do_flip:
                flipped = np.flipud(img.numpy())
                img_flipped = torch.from_numpy(_fix_negative_strides(flipped))
            else:
                img_flipped = img.clone()

        return img_flipped
