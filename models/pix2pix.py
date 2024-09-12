import numpy as np
import torch
from torch import nn
from utils.image import restore_image, split_image


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )
        if apply_batchnorm:
            self.conv.append(nn.BatchNorm2d(out_channels))
        self.conv.append(nn.LeakyReLU())

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        )
        if apply_dropout:
            self.conv.append(nn.Dropout())
        self.conv.append(nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class Pix2Pix_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.epochs_passed = 0
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(in_channels, features, apply_batchnorm=False),
            EncoderBlock(features, 2 * features),
            EncoderBlock(2 * features, 4 * features),
            EncoderBlock(4 * features, 8 * features),
            EncoderBlock(8 * features, 8 * features),
            EncoderBlock(8 * features, 8 * features),
            EncoderBlock(8 * features, 8 * features),
            EncoderBlock(8 * features, 8 * features, apply_batchnorm=False),
        )
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(8 * features, 8 * features, apply_dropout=True),
            DecoderBlock(8 * features + 8 * features, 8 * features, apply_dropout=True),
            DecoderBlock(8 * features + 8 * features, 8 * features, apply_dropout=True),
            DecoderBlock(8 * features + 8 * features, 8 * features),
            DecoderBlock(8 * features + 8 * features, 4 * features),
            DecoderBlock(4 * features + 256, 2 * features),
            DecoderBlock(2 * features + 2 * features, features),
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        skip_list = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_list.append(x)
        skip_list = reversed(skip_list[:-1])
        return x, skip_list

    def decode(self, x, skip_list):
        for block, skip in zip(self.decoder_blocks, skip_list):
            x = block(x)
            x = torch.cat([x, skip], dim=1)
        x = self.out(x)
        return x

    def forward(self, x):
        x, skip_list = self.encode(x)
        return self.decode(x, skip_list)

    def infere(self, img):
        # add them to batch
        patches = split_image(img, 256, 256)
        n_rows, n_cols = np.array(patches).shape[:2]
        generated_patches = []
        for i in range(n_rows):
            cur_row = []
            for j in range(n_cols):
                patch = torch.from_numpy(patches[i][j])
                patch = patch[None, :, :, :].cuda()

                gen_patch = torch.squeeze(self(patch))  # .permute(2, 1, 0)  # 1 2 0
                gen_patch = gen_patch.detach().cpu().numpy()
                cur_row.append(gen_patch)
            generated_patches.append(cur_row)

        generated_patches = np.array(generated_patches)
        restored = restore_image(generated_patches, *img.shape[-2:])
        return restored


class Pix2Pix_Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.epochs_passed = 0
        self.disc = nn.Sequential(
            EncoderBlock(in_channels + out_channels, features, apply_batchnorm=False),
            EncoderBlock(features, 2 * features),
            EncoderBlock(2 * features, 4 * features),
            nn.Conv2d(4 * features, 8 * features, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(8 * features),
            nn.LeakyReLU(),
            nn.Conv2d(8 * features, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.disc(x)
