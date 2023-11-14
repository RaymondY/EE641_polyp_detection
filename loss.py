# Author: Tiankai Yang <raymondyangtk@gmail.com>

import torch
import torch.nn as nn
from config import DefaultConfig

config = DefaultConfig()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum(axis=(1, 2, 3))
        union = pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean(axis=0)
