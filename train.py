# Author: Tiankai Yang <raymondyangtk@gmail.com>

import torch
import torch.optim as optim
from tqdm import tqdm
from loss import DiceLoss
from evaluate import evaluate, show_predicted_mask_example
from config import DefaultConfig

config = DefaultConfig()


def train_basic_unet(model, train_loader, test_loader=None):
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = DiceLoss()
    # criterion.to(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.train()
    for epoch in range(config.epoch_num):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask in tepoch:
                image = image.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                pred = model(image)
                loss = criterion(pred, mask)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        scheduler.step()
        print(f"Epoch {epoch + 1} training loss: {epoch_loss / len(train_loader)}")
        if test_loader:
            evaluate(model, train_loader)
            show_predicted_mask_example(model, train_loader)
            model.train()
    torch.save(model.state_dict(), config.model_path)
    print(f"Model saved to {config.model_path}")
