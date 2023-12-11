# Author: Tiankai Yang <raymondyangtk@gmail.com>

import torch
from utils import load_data, KvasirDataset, MultiScaleKvasirDataset
from model import (BasicUnet, MultiScaleUnet, MultiScalePixelShuffleUnet,
                   BasicBlock, ResBlock)
from train import train_basic_unet, train_multi_scale_unet
from evaluate import evaluate
from config import DefaultConfig

config = DefaultConfig()
model_class_set = {"Basic Unet": [KvasirDataset,
                                  BasicUnet,
                                  train_basic_unet],
                   "Multi-Scale Unet": [MultiScaleKvasirDataset,
                                        MultiScaleUnet,
                                        train_multi_scale_unet],
                   "Multi-Scale PixelShuffle Unet": [MultiScaleKvasirDataset,
                                                     MultiScalePixelShuffleUnet,
                                                     train_multi_scale_unet]}


def run(model_name, block_class):
    dataset_class, model_class, train_func = model_class_set[model_name]
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_class.__name__}")
    print(f"Model: {model_class.__name__}")
    print(f"Train function: {train_func.__name__}")

    # train example
    train_loader = load_data(is_train=True, dataset_class=dataset_class)
    test_loader = load_data(is_train=False, dataset_class=dataset_class)
    model = model_class(block_class)
    train_func(model, train_loader, test_loader)

    # # test example
    # model = model_class(block_class)
    # test_loader = load_data(is_train=False, dataset_class=dataset_class)
    # model.load_state_dict(torch.load("saved_models/xxx.pth"))
    # model.eval()
    # evaluate(model, test_loader)


def main():
    run(model_name="Basic Unet", block_class=BasicBlock)
    run(model_name="Multi-Scale Unet", block_class=BasicBlock)
    run(model_name="Multi-Scale Unet", block_class=ResBlock)
    run(model_name="Multi-Scale PixelShuffle Unet", block_class=BasicBlock)
    run(model_name="Multi-Scale PixelShuffle Unet", block_class=ResBlock)


if __name__ == '__main__':
    main()
