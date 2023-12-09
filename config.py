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
    ori_kvair_seg_image_dir = os.path.join(current_dir, 'datasets', 'Kvasir-SEG', 'images')
    ori_kvair_seg_masks_dir = os.path.join(current_dir, 'datasets', 'Kvasir-SEG', 'masks')
    kvair_seg_image_dir = os.path.join(current_dir, 'datasets', 'resized_kvasir_seg', 'images')
    kvair_seg_masks_dir = os.path.join(current_dir, 'datasets', 'resized_kvasir_seg', 'masks')
    train_ratio = 0.9

    # model dir
    model_dir = os.path.join(current_dir, 'saved_models')
    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pth'
    model_path = os.path.join(model_dir, model_name)

    # data
    width = 256
    height = 256

    # network
    init_features = 16

    # train
    epoch_num = 30
    warmup_epoch_num = 3
    lr = 2e-4
    warmup_lr = 1e-6
    batch_size = 4
    weight_decay = 0

    # # loss ratio
    # loss_ratio = 0.3

    threshold = 0.5
    smooth_factor = 1e-6
