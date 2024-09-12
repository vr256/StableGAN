import torch
from torch import nn

L1_loss = nn.L1Loss()
BCE_loss = nn.BCEWithLogitsLoss()


def wgan_generator_loss(y_gen, disc_y_gen, y_real, Lambda=100):
    return BCE_loss(disc_y_gen, torch.ones_like(disc_y_gen)) + Lambda * L1_loss(
        y_gen, y_real
    )


def wgan_discriminator_loss(disc_real, disc_y_gen):
    return (
        BCE_loss(disc_real, torch.ones_like(disc_real))
        + BCE_loss(disc_y_gen, torch.zeros_like(disc_y_gen))
    ) * 0.5


def pix2pix_generator_loss(y_gen, disc_y_gen, y_real, Lambda=100):
    return BCE_loss(disc_y_gen, torch.ones_like(disc_y_gen)) + Lambda * L1_loss(
        y_gen, y_real
    )


def pix2pix_discriminator_loss(disc_real, disc_y_gen):
    return (
        BCE_loss(disc_real, torch.ones_like(disc_real))
        + BCE_loss(disc_y_gen, torch.zeros_like(disc_y_gen))
    ) * 0.5
