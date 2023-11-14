# Author: Tiankai Yang <raymondyangtk@gmail.com>

import os
from datetime import datetime
import torch
import torch.optim as optim
from tqdm import tqdm
from loss import DiceLoss
from evaluate import evaluate, show_predicted_mask_example, show_multi_scale_predicted_mask_example
from config import DefaultConfig

config = DefaultConfig()


def train_basic_unet(model, train_loader, test_loader=None):
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25, 28], gamma=0.1)
    warmup_optimizer = optim.Adam(model.parameters(), lr=config.warmup_lr)
    criterion = torch.nn.BCELoss()
    # criterion.to(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.train()
    # warm up training
    print("Warm up training...")
    for epoch in range(config.warmup_epoch_num):
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask in tepoch:
                loss = _train_basic_unet_helper(image, mask,
                                                warmup_optimizer, model, criterion, device)
                tepoch.set_postfix(loss=loss)
    print("Warm up training finished.")
    # normal training
    for epoch in range(config.epoch_num):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask in tepoch:
                temp_loss = _train_basic_unet_helper(image, mask,
                                                     optimizer, model, criterion, device)
                epoch_loss += temp_loss
                tepoch.set_postfix(loss=temp_loss)

        scheduler.step()
        print(f"Epoch {epoch + 1} training loss: {epoch_loss / len(train_loader)}")
        if test_loader:
            evaluate(model, test_loader)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                show_predicted_mask_example(model, test_loader)
            model.train()
    model_path = os.path.join(config.model_dir,
                              f"basic_unet_{datetime.now().strftime('%m%d%H%M')}.pth")
    torch.save(model.state_dict(), model_path)


def _train_basic_unet_helper(image, mask, optimizer, model, criterion, device):
    image = image.to(device)
    mask = mask.to(device)
    optimizer.zero_grad()
    pred = model(image)
    loss = criterion(pred, mask)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_multi_scale_unet(model, train_loader, test_loader=None):
    device = config.device
    coarse_optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    fine_optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    coarse_scheduler = optim.lr_scheduler.MultiStepLR(coarse_optimizer, milestones=[20, 25, 28], gamma=0.1)
    fine_scheduler = optim.lr_scheduler.MultiStepLR(fine_optimizer, milestones=[20, 25, 28], gamma=0.1)
    warmup_coarse_optimizer = optim.Adam(model.parameters(), lr=config.warmup_lr)
    warmup_fine_optimizer = optim.Adam(model.parameters(), lr=config.warmup_lr)
    criterion = DiceLoss()
    # criterion = torch.nn.BCELoss()
    # criterion.to(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.train()

    # warm up training
    print("Warm up training...")
    for epoch in range(config.warmup_epoch_num):
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask1, mask2, mask3, mask4 in tepoch:
                temp_coarse_loss, temp_fine_loss = \
                    _train_multi_scale_unet_helper(image, mask1, mask2, mask3, mask4,
                                                   warmup_coarse_optimizer,
                                                   warmup_fine_optimizer,
                                                   model, criterion, device)
                tepoch.set_postfix(loss4=temp_coarse_loss, loss1=temp_fine_loss)
    print("Warm up training finished.")

    # normal training
    for epoch in range(config.epoch_num):
        epoch_coarse_loss = 0.0
        epoch_fine_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask1, mask2, mask3, mask4 in tepoch:
                coarse_loss, fine_loss = _train_multi_scale_unet_helper(image, mask1, mask2, mask3, mask4,
                                                                        coarse_optimizer, fine_optimizer,
                                                                        model, criterion, device)
                epoch_coarse_loss += coarse_loss
                epoch_fine_loss += fine_loss
                tepoch.set_postfix(loss4=coarse_loss, loss1=fine_loss, lr=fine_optimizer.param_groups[0]['lr'])
        coarse_scheduler.step()
        fine_scheduler.step()
        print(f"Epoch {epoch + 1} training coarse loss: {epoch_coarse_loss / len(train_loader)}")
        print(f"Epoch {epoch + 1} training fine loss: {epoch_fine_loss / len(train_loader)}")
        if test_loader:
            evaluate(model, test_loader)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                show_multi_scale_predicted_mask_example(model, test_loader)
            model.train()
    model_path = os.path.join(config.model_dir,
                              f"multi_scale_unet_{datetime.now().strftime('%m%d%H%M')}.pth")
    torch.save(model.state_dict(), model_path)


def _train_multi_scale_unet_helper(image, mask1, mask2, mask3, mask4,
                                   coarse_optimizer, fine_optimizer,
                                   model, criterion, device):
    image = image.to(device)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)
    mask3 = mask3.to(device)
    mask4 = mask4.to(device)

    # level 4
    coarse_optimizer.zero_grad()
    enc4 = model.forward_encoder(image, need_level=4)
    dec4, pred4 = model.forward_level_4(enc4)
    coarse_loss = criterion(pred4, mask4)
    coarse_loss.backward()
    coarse_optimizer.step()

    # level 1
    fine_optimizer.zero_grad()
    enc1, enc2, enc3, enc4 = model.forward_encoder(image, need_level=1)
    pred1, pred2, pred3, pred4 = model.forward_level_1(enc1, enc2, enc3, enc4)
    fine_loss = (criterion(pred1, mask1) +
                 criterion(pred2, mask2) +
                 criterion(pred3, mask3) +
                 criterion(pred4, mask4))
    fine_loss.backward()
    fine_optimizer.step()

    return coarse_loss.item(), fine_loss.item()
    # return 0, fine_loss.item()
