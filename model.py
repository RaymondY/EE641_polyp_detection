# Author: Tiankai Yang <raymondyangtk@gmail.com>

import torch
import torch.nn as nn
from config import DefaultConfig

config = DefaultConfig()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 has_act=True, activation='relu', has_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.has_act = has_act
        self.activation = None
        if has_act:
            self._set_activation(activation)
        self.has_bn = has_bn
        if has_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def _set_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Activation {activation} is not supported.")

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.activation(x)
        return x


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BasicUnet(nn.Module):
    def __init__(self, unet_block_module=UnetBlock, in_channels=3, out_channels=1, init_features=32):
        super(BasicUnet, self).__init__()
        features = init_features
        self.encoder1 = unet_block_module(in_channels, features)
        self.encoder2 = unet_block_module(features, features * 2)
        self.encoder3 = unet_block_module(features * 2, features * 4)
        self.encoder4 = unet_block_module(features * 4, features * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = unet_block_module(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = unet_block_module((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = unet_block_module((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = unet_block_module((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = unet_block_module(features * 2, features)
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.conv(dec1)
        # output = self.sigmoid(output)

        return output


class MultiScaleUnet(BasicUnet):
    def __init__(self, unet_block_module=UnetBlock, in_channels=3, out_channels=1, init_features=32):
        super(MultiScaleUnet, self).__init__(unet_block_module, in_channels, out_channels, init_features)
        features = init_features
        self.decoder3 = unet_block_module((features * 4) * 2 + 1, features * 4)
        self.decoder2 = unet_block_module((features * 2) * 2 + 1, features * 2)
        self.decoder1 = unet_block_module(features * 2 + 1, features)
        self.conv4 = nn.Conv2d(
            in_channels=init_features * 8, out_channels=out_channels, kernel_size=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=init_features * 4, out_channels=out_channels, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=init_features * 2, out_channels=out_channels, kernel_size=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=init_features, out_channels=out_channels, kernel_size=1
        )
        self.upconvformask4 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
        self.upconvformask3 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
        self.upconvformask2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )

    def forward_encoder(self, x, need_level=None):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        if need_level == 1:
            return enc1, enc2, enc3, enc4
        elif need_level == 2:
            return enc2, enc3, enc4
        elif need_level == 3:
            return enc3, enc4
        elif need_level == 4:
            return enc4
        else:
            return enc1, enc2, enc3, enc4

    def forward_level_4(self, enc4):
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        pred4 = self.conv4(dec4)
        return dec4, pred4

    def forward_level_3(self, enc3, enc4):
        dec4, pred4 = self.forward_level_4(enc4)
        dec3 = self.upconv3(dec4)
        pred3 = self.upconvformask4(pred4)
        dec3 = torch.cat((pred3, dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        pred3 = self.conv3(dec3)
        return dec3, pred3, pred4

    def forward_level_2(self, enc2, enc3, enc4):
        dec3, pred3, pred4 = self.forward_level_3(enc3, enc4)
        dec2 = self.upconv2(dec3)
        pred2 = self.upconvformask3(pred3)
        dec2 = torch.cat((pred2, dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        pred2 = self.conv2(dec2)
        return dec2, pred2, pred3, pred4

    def forward_level_1(self, enc1, enc2, enc3, enc4):
        dec2, pred2, pred3, pred4 = self.forward_level_2(enc2, enc3, enc4)
        dec1 = self.upconv1(dec2)
        pred1 = self.upconvformask2(pred2)
        dec1 = torch.cat((pred1, dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        pred1 = self.conv1(dec1)
        return pred1, pred2, pred3, pred4

    def forward(self, x):
        # Encoder
        enc1, enc2, enc3, enc4 = self.forward_encoder(x)
        # Decoder
        pred1 = self.forward_level_1(enc1, enc2, enc3, enc4)[0]
        return pred1


