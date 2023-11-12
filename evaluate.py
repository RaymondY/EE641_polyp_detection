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
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion.to(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for image, mask in test_loader:
            image = image.to(device)
            mask = mask.to(device)
            pred = model(image)
            loss = criterion(pred, mask)
            epoch_loss += loss.item()
    print(f"Test loss: {epoch_loss / len(test_loader)}")


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
    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask[0].cpu().squeeze())
    plt.subplot(1, 3, 3)
    plt.title("Predicted mask")
    plt.imshow(pred[0].detach().cpu().squeeze())
    plt.show()
