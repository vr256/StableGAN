import random
from enum import Enum, auto

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BRIGTHNESS_LOW = 0.25
BRIGTHNESS_HIGH = 1.25
NOISE_LOW = 0
NOISE_HIGH = 1.25
NOISE_MEAN = 0.5
NOISE_STDDEV = 1


class Defect(Enum):
    Brightness = auto()
    Blur = auto()
    Noise = auto()


def randomly_degrade(tensor):
    defects = [Defect.Brightness, Defect.Blur, Defect.Noise]
    original_defect = random.choice(defects)
    for defect in defects:
        if defect is not original_defect:
            is_present = np.random.rand() < 0.5
        else:
            is_present = True

        if is_present:
            match defect:
                case Defect.Brightness:
                    tensor = degrade_brightness(tensor)
                case Defect.Blur:
                    tensor = apply_blur(tensor)
                case Defect.Noise:
                    tensor = add_noise(tensor)

    return torch.clip(tensor, 0, 1)


def add_noise(tensor, noise_coef=None):
    noise_coefs = [
        np.random.uniform(NOISE_LOW, NOISE_HIGH),
        np.random.normal(NOISE_MEAN, NOISE_STDDEV),
    ]
    if noise_coef is None:
        noise_coef = random.choice(noise_coefs)
    noise = torch.rand_like(tensor) * noise_coef
    return tensor + noise


def apply_blur(tensor, kernel_size=None):
    tensor_np = tensor.permute(1, 2, 0).cpu().numpy()
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7, 9, 11, 13])
    tensor_np = cv2.GaussianBlur(tensor_np, (kernel_size, kernel_size), 0)
    tensor = torch.tensor(tensor_np).permute(2, 0, 1).float()
    return tensor


def degrade_brightness(tensor, brightness=None):
    if brightness is None:
        brightness = np.random.uniform(BRIGTHNESS_LOW, BRIGTHNESS_HIGH)
    tensor = tensor - (1 - brightness)
    return tensor


def box_filter(tensor, kernel_size=3):
    n_channels = tensor.shape[-3]
    # output image should be of the same size as input one
    # therefore may need to check different kernels
    kernels = range(min(kernel_size, 3), kernel_size + 4)
    neares_kernels = sorted(kernels, key=lambda x: abs(x - kernel_size))

    for kernel_size in neares_kernels:
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        kernel = kernel / kernel.sum()

        img = F.conv2d(
            tensor,
            kernel.repeat(n_channels, 1, 1, 1),
            padding=(kernel_size - 1) // 2,
            groups=n_channels,
        )
        if img.shape == tensor.shape:
            break

    return img


def gaussian_filter(tensor, kernel_size=3, std=1):
    n_channels = tensor.shape[-3]
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    y = x.view(-1, 1)
    kernel = torch.exp(-(x**2 + y**2) / (2 * std**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    return F.conv2d(
        tensor,
        kernel.repeat(n_channels, 1, 1, 1),
        padding=(kernel_size - 1) // 2,
        groups=n_channels,
    )
