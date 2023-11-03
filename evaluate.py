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
    with torch.no_grad():
        image, mask = test_loader.dataset[index]
        image = image.to(device)
        mask = mask.to(device)
        pred = model(image.unsqueeze(0)).squeeze(0)
        # pred = (pred > 0.5).float()
        pred = pred.cpu().numpy()
        pred = pred.transpose(1, 2, 0)
        mask = mask.cpu().numpy()
        mask = mask.transpose(1, 2, 0)
        image = denormalize(image)
        image = image.cpu().numpy()
        image = image.transpose(1, 2, 0)

        plt.subplot(131)
        plt.imshow(image)
        plt.subplot(132)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.subplot(133)
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.show()
