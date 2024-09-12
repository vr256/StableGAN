import os
from dataclasses import dataclass, field
from enum import Enum
from glob import glob

import torch
from models import (
    Pix2Pix_Discriminator,
    Pix2Pix_Generator,
    WGAN_Discriminator,
    WGAN_Generator,
)
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from training.losses import (
    pix2pix_discriminator_loss,
    pix2pix_generator_loss,
    wgan_discriminator_loss,
    wgan_generator_loss,
)


@dataclass
class Config:
    # Hyperparameters
    learning_rate: float = field(default=2 * 10e-4)
    beta1: float = field(default=0.5)
    beta2: float = field(default=0.999)

    # Model-related
    generator: torch.nn.Module = field(default=None)
    discriminator: torch.nn.Module = field(default=None)
    generator_loss: callable = field(default=None)
    discriminator_loss: callable = field(default=None)
    generator_opt: torch.optim.Optimizer = field(default=None)
    discriminator_opt: torch.optim.Optimizer = field(default=None)

    # Misc
    batch_size: int = field(default=32)
    image_size: int = field(default=256)
    dataset_name: str = field(default="")
    root: str = field(default="")


@dataclass
class TestPerformanceReport:
    fid: float = field(default=0)
    kid: float = field(default=0)
    accuracy: float = field(default=0)

    def __repr__(self):
        return f"FID score: {self.fid:.2f}\n"  # f"KID score: {self.kid:.4f}\n"


class CustomDataset:
    """Unstructured dataset for training WGANs"""

    def __init__(self, root=None, mode="train"):
        self.mode = mode
        self.root = root

        assert mode in ["train", "test"], "Mode must be either 'train' or 'test'."

        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if type(self) is CustomDataset:
            self.files = sorted(glob(os.path.join(root, f"{mode}") + "/*.*"))
            self.n_images = len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = self.transforms(image)
        return image

    def __len__(self):
        return self.n_images


class PairedDataset(CustomDataset):
    """Structured dataset for training Pix2Pix GANs"""

    def __init__(self, root=None, mode="train"):
        super().__init__(root, mode)

        self.files_input = sorted(glob(os.path.join(root, f"{mode}_input") + "/*.*"))
        self.files_output = sorted(glob(os.path.join(root, f"{mode}_output") + "/*.*"))
        self.n_images = len(self.files_input)

        assert len(self.files_input) == len(
            self.files_output
        ), "The number of input and output files must be the same."

    def __getitem__(self, idx):
        image_input = Image.open(self.files_input[idx])
        image_output = Image.open(self.files_output[idx])

        image_input = self.transforms(image_input)
        image_output = self.transforms(image_output)

        return image_input, image_output


class DatasetLoader(Enum):
    AERIAL_DEGRADED_256 = (
        "/home/vr256/Desktop/Data/Images/structured/degraded_combined_aerial/256"
    )
    AERIAL_DEGRADED_512 = (
        "/home/vr256/Desktop/Data/Images/structured/degraded_combined_aerial/512"
    )
    FACADE = "/home/vr256/Desktop/Data/Images/structured/facade"
    AERIAL_RAW_256 = "/home/vr256/Desktop/Data/Images/structured/combined_aerial/256"
    AERIAL_RAW_512 = "/home/vr256/Desktop/Data/Images/structured/combined_aerial/512"
    BEDROOMS = "/home/vr256/Desktop/Data/Images/structured/bedrooms"

    def load(self, config: Config, extra_dl=False):
        """Load dataset into torch DataLoaders"""
        self.setup_config(config)
        match self:
            case (
                DatasetLoader.AERIAL_DEGRADED_256
                | DatasetLoader.AERIAL_DEGRADED_512
                | DatasetLoader.FACADE
            ):
                dataset = PairedDataset
            case (
                DatasetLoader.AERIAL_RAW_256
                | DatasetLoader.AERIAL_RAW_512
                | DatasetLoader.BEDROOMS
            ):
                dataset = CustomDataset

        dataset_train = dataset(root=self.value, mode="train")
        dataset_test = dataset(root=self.value, mode="test")

        train_dl = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        test_dl = DataLoader(
            dataset_test,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        if extra_dl:
            extra_dl = DataLoader(
                dataset_train,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
            )

        return (train_dl, extra_dl, test_dl) if extra_dl else (train_dl, test_dl)

    def setup_config(self, config: Config):
        """Set up models and loss functions depending on a chosen dataset"""
        config.dataset_name = self.name
        match self:
            case DatasetLoader.AERIAL_DEGRADED_256 | DatasetLoader.FACADE:
                # Pix2Pix
                config.generator = Pix2Pix_Generator
                config.discriminator = Pix2Pix_Discriminator
                config.generator_loss = pix2pix_generator_loss
                config.discriminator_loss = pix2pix_discriminator_loss
                config.image_size = 256
            case DatasetLoader.AERIAL_RAW_256 | DatasetLoader.BEDROOMS:
                # WGAN
                config.generator = WGAN_Generator
                config.discriminator = WGAN_Discriminator
                config.generator_loss = wgan_generator_loss
                config.discriminator_loss = wgan_discriminator_loss
                config.image_size = 256
            case DatasetLoader.AERIAL_DEGRADED_512:
                # Pix2Pix
                config.generator = Pix2Pix_Generator
                config.discriminator = Pix2Pix_Discriminator
                config.generator_loss = pix2pix_generator_loss
                config.discriminator_loss = pix2pix_discriminator_loss
                config.image_size = 512
            case DatasetLoader.AERIAL_RAW_512:
                # WGAN
                config.generator = WGAN_Generator
                config.discriminator = WGAN_Discriminator
                config.generator_loss = wgan_generator_loss
                config.discriminator_loss = wgan_discriminator_loss
                config.image_size = 512
