# Author: Tiankai Yang <raymondyangtk@gmail.com>

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from loss import DiceLoss
from utils import denormalize
from config import DefaultConfig

config = DefaultConfig()


def evaluate(model, test_loader):
    device = config.device
    criterion = DiceLoss()
    # criterion = torch.nn.BCELoss()
    # criterion.to(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()
    dice_loss = 0.0
    iou_loss = 0.0
    dice_sum = 0.0
    with torch.no_grad():
        for image, mask in test_loader:
            image = image.to(device)
            mask = mask.to(device)
            pred = model(image)
            loss = criterion(pred, mask)
            dice_loss += loss.item()
            pred = (pred > config.threshold).float()
            dice_sum += compute_dice(pred, mask)
    print(f"Test Dice Loss: {dice_loss / len(test_loader)}")
    print(f"Test IoU Loss: {iou_loss / len(test_loader)}")
    print(f"Test mDice: {dice_sum / len(test_loader)}")


def compute_iou(pred, mask):

    intersection = (pred * mask).sum(axis=(1, 2, 3))
    union = pred.sum(axis=(1, 2, 3)) + mask.sum(axis=(1, 2, 3)) - intersection
    iou = (intersection + config.smooth_factor) / (union + config.smooth_factor)
    return iou.mean(axis=0)


def compute_dice(pred, mask):
    intersection = (pred * mask).sum(axis=(1, 2, 3))
    union = pred.sum(axis=(1, 2, 3)) + mask.sum(axis=(1, 2, 3))
    dice = (2 * intersection + config.smooth_factor) / (union + config.smooth_factor)
    return dice.mean(axis=0)


def show_predicted_mask_example(model, test_loader, index=0):
    device = config.device
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()
    image, mask = test_loader.dataset[index]
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    pred = model(image)
    image = denormalize(image, device)
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image[0].permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Real mask")
    plt.imshow(mask[0].cpu().squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Pred mask")
    plt.imshow(pred[0].detach().cpu().squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()


def show_multi_scale_predicted_mask_example(model, test_loader, index=0):
    device = config.device
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()
    image, mask = test_loader.dataset[index]
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    pred1, pred2, pred3, pred4 = model(image, need_all_levels=True)
    image = denormalize(image, device)
    plt.subplot(2, 3, 1)
    plt.title("Image")
    plt.imshow(image[0].permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title("Real mask")
    plt.imshow(mask[0].cpu().squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.title("Pred mask level 4")
    plt.imshow(pred4[0].detach().cpu().squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.title("Pred mask level 3")
    plt.imshow(pred3[0].detach().cpu().squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title("Pred mask level 2")
    plt.imshow(pred2[0].detach().cpu().squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.title("Pred mask level 1")
    plt.imshow(pred1[0].detach().cpu().squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
