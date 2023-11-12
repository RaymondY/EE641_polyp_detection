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
    # criterion = torch.nn.BCEWithLogitsLoss()
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
            evaluate(model, test_loader)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                show_predicted_mask_example(model, test_loader)
            model.train()
    torch.save(model.state_dict(), config.model_path)
    print(f"Model saved to {config.model_path}")


def train_multi_scale_unet(model, train_loader, test_loader=None):
    device = config.device
    optimizer4 = optim.Adam(model.parameters(), lr=config.lr)
    optimizer3 = optim.Adam(model.parameters(), lr=config.lr)
    optimizer2 = optim.Adam(model.parameters(), lr=config.lr)
    optimizer1 = optim.Adam(model.parameters(), lr=config.lr)
    scheduler4 = optim.lr_scheduler.StepLR(optimizer4, step_size=5, gamma=0.3)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=5, gamma=0.3)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.3)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.3)
    criterion = DiceLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion.to(device)
    if next(model.parameters()).device != device:
        model.to(device)
    model.train()

    for epoch in range(config.epoch_num):
        epoch_loss4, epoch_loss3, epoch_loss2, epoch_loss1 = 0.0, 0.0, 0.0, 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for image, mask1, mask2, mask3, mask4 in tepoch:
                image = image.to(device)
                mask1 = mask1.to(device)
                mask2 = mask2.to(device)
                mask3 = mask3.to(device)
                mask4 = mask4.to(device)

                # level 4
                optimizer4.zero_grad()
                enc4 = model.forward_encoder(image, need_level=4)
                dec4, pred4 = model.forward_level_4(enc4)
                loss4 = criterion(pred4, mask4)
                loss4.backward()
                optimizer4.step()
                epoch_loss4 += loss4.item()

                # level 3
                optimizer3.zero_grad()
                enc3, enc4 = model.forward_encoder(image, need_level=3)
                dec3, pred3, pred4 = model.forward_level_3(enc3, enc4)
                loss3 = criterion(pred3, mask3) + \
                    config.loss_ratio * criterion(pred4, mask4)
                loss3.backward()
                optimizer3.step()
                epoch_loss3 += loss3.item()

                # level 2
                optimizer2.zero_grad()
                enc2, enc3, enc4 = model.forward_encoder(image, need_level=2)
                dec2, pred2, pred3, pred4 = model.forward_level_2(enc2, enc3, enc4)
                loss2 = criterion(pred2, mask2) + \
                    config.loss_ratio * criterion(pred3, mask3) + \
                    config.loss_ratio * criterion(pred4, mask4)
                loss2.backward()
                optimizer2.step()
                epoch_loss2 += loss2.item()

                # level 1
                optimizer1.zero_grad()
                enc1, enc2, enc3, enc4 = model.forward_encoder(image, need_level=1)
                pred1, pred2, pred3, pred4 = model.forward_level_1(enc1, enc2, enc3, enc4)
                loss1 = criterion(pred1, mask1) + \
                    config.loss_ratio * criterion(pred2, mask2) + \
                    config.loss_ratio * criterion(pred3, mask3) + \
                    config.loss_ratio * criterion(pred4, mask4)
                loss1.backward()
                optimizer1.step()
                epoch_loss1 += loss1.item()

                tepoch.set_postfix(loss4=loss4.item(), loss3=loss3.item(),
                                   loss2=loss2.item(), loss1=loss1.item())
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
                show_predicted_mask_example(model, test_loader)
            model.train()
    torch.save(model.state_dict(), config.model_path)
    print(f"Model saved to {config.model_path}")
