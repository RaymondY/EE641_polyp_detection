# Author: Tiankai Yang <raymondyangtk@gmail.com>

from utils import load_data, KvasirDataset, MultiScaleKvasirDataset
from model import BasicUnet, MultiScaleUnet
from train import train_basic_unet, train_multi_scale_unet
from config import DefaultConfig

config = DefaultConfig()
model_class_set = {"Basic Unet": [KvasirDataset,
                                  BasicUnet,
                                  train_basic_unet],
                   "Multi-Scale Unet": [MultiScaleKvasirDataset,
                                        MultiScaleUnet,
                                        train_multi_scale_unet]}


def run(model_name="Basic Unet"):
    dataset_class, model_class, train_func = model_class_set[model_name]
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_class.__name__}")
    print(f"Model: {model_class.__name__}")
    print(f"Train function: {train_func.__name__}")
    train_loader = load_data(is_train=True, dataset_class=dataset_class)
    test_loader = load_data(is_train=False, dataset_class=dataset_class)
    model = model_class()
    train_func(model, train_loader, test_loader)


def main():
    # run(model_name="Basic Unet")
    run(model_name="Multi-Scale Unet")


if __name__ == '__main__':
    main()
