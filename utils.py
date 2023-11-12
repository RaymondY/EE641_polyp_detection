# Author: Tiankai Yang <raymondyangtk@gmail.com>

import os
from PIL import Image
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


class MultiScaleKvasirDataset(KvasirDataset):
    def __init__(self, image_dir, mask_dir,
                 image_transform=None, mask_transform=None,
                 is_train=True):
        super(MultiScaleKvasirDataset, self).__init__(image_dir, mask_dir,
                                                      image_transform, mask_transform,
                                                      is_train)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        mask_path = self.mask_path_list[index]
        image = Image.open(image_path).convert('RGB')
        mask1 = Image.open(mask_path).convert('L')
        mask2, mask3, mask4 = None, None, None
        if self.is_train:
            # use Lanczos to resize mask
            mask2 = mask1.resize((config.width // 2, config.height // 2), resample=Image.LANCZOS)
            mask3 = mask1.resize((config.width // 4, config.height // 4), resample=Image.LANCZOS)
            mask4 = mask1.resize((config.width // 8, config.height // 8), resample=Image.LANCZOS)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask1 = self.mask_transform(mask1)
            if self.is_train:
                mask2 = self.mask_transform(mask2)
                mask3 = self.mask_transform(mask3)
                mask4 = self.mask_transform(mask4)
        if self.is_train:
            return image, mask1, mask2, mask3, mask4
        else:
            return image, mask1


def load_data(is_train=True, dataset_class=KvasirDataset):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5592, 0.3213, 0.2349],
                             [0.3057, 0.2139, 0.1766])
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = dataset_class(config.kvair_seg_image_dir, config.kvair_seg_masks_dir,
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


def denormalize(image, device="cpu"):
    mean = torch.tensor([0.5592, 0.3213, 0.2349]).view(3, 1, 1).to(device)
    std = torch.tensor([0.3057, 0.2139, 0.1766]).view(3, 1, 1).to(device)
    return image * std + mean


if __name__ == '__main__':
    compute_mean_std()
