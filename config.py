# Author: Tiankai Yang <raymondyangtk@gmail.com>

import os
import torch
from datetime import datetime


class DefaultConfig:
    # general
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    # datasets dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ori_kvair_seg_image_dir = os.path.join(current_dir, 'datasets', 'kvasir', 'Kvasir-SEG', 'images')
    ori_kvair_seg_masks_dir = os.path.join(current_dir, 'datasets', 'kvasir', 'Kvasir-SEG', 'masks')
    kvair_seg_image_dir = os.path.join(current_dir, 'datasets', 'normalized_kvasir_seg', 'images')
    kvair_seg_masks_dir = os.path.join(current_dir, 'datasets', 'normalized_kvasir_seg', 'masks')
    train_ratio = 0.7

    # model dir
    model_dir = os.path.join(current_dir, 'saved_models')
    model_path = 'model_{}.pth'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

    # data
    width = 640
    height = 640

    # network
    epoch_num = 100
    lr = 1e-3
    batch_size = 16
