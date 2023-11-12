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
    # optimizer = optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=1e-8,
    #                           momentum=0.9, foreach=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    warmup_optimizer = optim.Adam(model.parameters(), lr=config.warmup_lr)
    criterion = DiceLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
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
    optimizer4 = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer3 = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer2 = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer1 = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler4 = optim.lr_scheduler.StepLR(optimizer4, step_size=8, gamma=0.2)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=8, gamma=0.2)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=8, gamma=0.2)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=8, gamma=0.2)
    warmup_optimizer4 = optim.Adam(model.parameters(), lr=config.warmup_lr)
    warmup_optimizer3 = optim.Adam(model.parameters(), lr=config.warmup_lr)
    warmup_optimizer2 = optim.Adam(model.parameters(), lr=config.warmup_lr)
    warmup_optimizer1 = optim.Adam(model.parameters(), lr=config.warmup_lr)
    criterion = DiceLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion.to(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.train()

    # warm up training
    print("Warm up training...")
    for epoch in range(config.warmup_epoch_num):
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask1, mask2, mask3, mask4 in tepoch:
                loss4, loss3, loss2, loss1 = \
                    _train_multi_scale_unet_helper(image, mask1, mask2, mask3, mask4,
                                                   warmup_optimizer4, warmup_optimizer3,
                                                   warmup_optimizer2, warmup_optimizer1,
                                                   model, criterion, device)
                tepoch.set_postfix(loss4=loss4, loss3=loss3, loss2=loss2, loss1=loss1)
    print("Warm up training finished.")

    # normal training
    for epoch in range(config.epoch_num):
        epoch_loss4, epoch_loss3, epoch_loss2, epoch_loss1 = 0.0, 0.0, 0.0, 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask1, mask2, mask3, mask4 in tepoch:
                # loss1 = _train_multi_scale_unet_helper(image, mask1, mask2, mask3, mask4,
                #                                        optimizer4, optimizer3, optimizer2, optimizer1,
                #                                        model, criterion, device)
                # epoch_loss1 += loss1
                # tepoch.set_postfix(loss1=loss1)
                loss4, loss3, loss2, loss1 = \
                    _train_multi_scale_unet_helper(image, mask1, mask2, mask3, mask4,
                                                   optimizer4, optimizer3, optimizer2, optimizer1,
                                                   model, criterion, device)
                epoch_loss4 += loss4
                epoch_loss3 += loss3
                epoch_loss2 += loss2
                epoch_loss1 += loss1
                tepoch.set_postfix(loss4=loss4, loss3=loss3, loss2=loss2, loss1=loss1)
        scheduler4.step()
        scheduler3.step()
        scheduler2.step()
        scheduler1.step()
        print(f"Epoch {epoch + 1} training loss4: {epoch_loss4 / len(train_loader)}")
        print(f"Epoch {epoch + 1} training loss3: {epoch_loss3 / len(train_loader)}")
        print(f"Epoch {epoch + 1} training loss2: {epoch_loss2 / len(train_loader)}")
        print(f"Epoch {epoch + 1} training loss1: {epoch_loss1 / len(train_loader)}")
        if test_loader:
            evaluate(model, test_loader)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                show_multi_scale_predicted_mask_example(model, test_loader)
            model.train()
    model_path = os.path.join(config.model_dir,
                              f"multi_scale_unet_{datetime.now().strftime('%m%d%H%M')}.pth")
    torch.save(model.state_dict(), model_path)


def _train_multi_scale_unet_helper(image, mask1, mask2, mask3, mask4,
                                   optimizer4, optimizer3, optimizer2, optimizer1,
                                   model, criterion, device):
    image = image.to(device)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)
    mask3 = mask3.to(device)
    mask4 = mask4.to(device)

    # optimizer1.zero_grad()
    # pred1, pred2, pred3, pred4 = model(image, need_all_levels=True)
    # loss = criterion(pred1, mask1) + \
    #     config.loss_ratio * criterion(pred2, mask2) + \
    #     config.loss_ratio * criterion(pred3, mask3) + \
    #     config.loss_ratio * criterion(pred4, mask4)
    # loss.backward()
    # optimizer1.step()
    # return loss.item()

    # level 4
    optimizer4.zero_grad()
    enc4 = model.forward_encoder(image, need_level=4)
    dec4, pred4 = model.forward_level_4(enc4)
    loss4 = criterion(pred4, mask4)
    loss4.backward()
    optimizer4.step()

    # level 3
    optimizer3.zero_grad()
    enc3, enc4 = model.forward_encoder(image, need_level=3)
    dec3, pred3, pred4 = model.forward_level_3(enc3, enc4)
    loss3 = criterion(pred3, mask3) + \
        config.loss_ratio * criterion(pred4, mask4)
    loss3.backward()
    optimizer3.step()

    # level 2
    optimizer2.zero_grad()
    enc2, enc3, enc4 = model.forward_encoder(image, need_level=2)
    dec2, pred2, pred3, pred4 = model.forward_level_2(enc2, enc3, enc4)
    loss2 = criterion(pred2, mask2) + \
        config.loss_ratio / 2 * criterion(pred3, mask3) + \
        config.loss_ratio / 2 * criterion(pred4, mask4)
    loss2.backward()
    optimizer2.step()

    # level 1
    optimizer1.zero_grad()
    enc1, enc2, enc3, enc4 = model.forward_encoder(image, need_level=1)
    pred1, pred2, pred3, pred4 = model.forward_level_1(enc1, enc2, enc3, enc4)
    loss1 = criterion(pred1, mask1) + \
        config.loss_ratio / 3 * criterion(pred2, mask2) + \
        config.loss_ratio / 3 * criterion(pred3, mask3) + \
        config.loss_ratio / 3 * criterion(pred4, mask4)
    loss1.backward()
    optimizer1.step()

    return loss4.item(), loss3.item(), loss2.item(), loss1.item()
