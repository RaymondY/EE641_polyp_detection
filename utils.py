# Author: Tiankai Yang <raymondyangtk@gmail.com>

import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import DefaultConfig

config = DefaultConfig()
train_ratio = config.train_ratio


class KvasirDataset(Dataset):
    def __init__(self, image_dir, mask_dir,
                 image_transform=None, mask_transform=None,
                 is_train=True):
        super(KvasirDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.is_train = is_train
        self.image_path_list = []
        self.mask_path_list = []
        self._set_files()

    def _set_files(self):
        name_list = os.listdir(self.image_dir)
        if self.is_train:
            name_list = name_list[:int(len(name_list) * train_ratio)]
        else:
            name_list = name_list[int(len(name_list) * train_ratio):]

        for name in name_list:
            image_path = os.path.join(self.image_dir, name)
            mask_path = os.path.join(self.mask_dir, name)
            self.image_path_list.append(image_path)
            self.mask_path_list.append(mask_path)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        mask_path = self.mask_path_list[index]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.image_path_list)


def load_data(is_train=True):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5592, 0.3213, 0.2349],
                             [0.3057, 0.2139, 0.1766])
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = KvasirDataset(config.kvair_seg_image_dir, config.kvair_seg_masks_dir,
                            image_transform=image_transform, mask_transform=mask_transform,
                            is_train=is_train)
    if is_train:
        shuffle = True
    else:
        shuffle = False
    data_loader = DataLoader(dataset, batch_size=config.batch_size,
                             shuffle=shuffle)
    return data_loader


def compute_mean_std():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = KvasirDataset(config.kvair_seg_image_dir, config.kvair_seg_masks_dir,
                            image_transform=transform, mask_transform=transform,
                            is_train=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for image, _ in data_loader:
        mean += torch.mean(image, dim=[0, 2, 3])
        std += torch.std(image, dim=[0, 2, 3])
    mean /= len(data_loader)
    std /= len(data_loader)
    print(mean, std)
    # tensor([0.5592, 0.3213, 0.2349]) tensor([0.3057, 0.2139, 0.1766])


def denormalize(image):
    mean = torch.tensor([0.5592, 0.3213, 0.2349]).view(3, 1, 1)
    std = torch.tensor([0.3057, 0.2139, 0.1766]).view(3, 1, 1)
    return image * std + mean


if __name__ == '__main__':
    compute_mean_std()
    # train_data_loader = load_data(is_train=True)
    # for image, mask in train_data_loader:
    #     print(image.shape, mask.shape)
    #     plt.imshow(image[0].permute(1, 2, 0))
    #     plt.show()
    #     plt.imshow(denormalize(image[0]).permute(1, 2, 0))
    #     plt.show()
    #     plt.imshow(mask[0].squeeze(), cmap='gray')
    #     plt.show()
    #     break
