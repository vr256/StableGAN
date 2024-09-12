import base64
import io
import os
import shutil
from glob import glob

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from PIL import Image
from preprocessing.degradation import randomly_degrade
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

from .image import read_image, split_image


def make_dirs(destination, sub_dirs=None):
    """Build a dir tree for further train-test split"""
    os.makedirs(destination, exist_ok=True)
    shutil.rmtree(destination)
    os.mkdir(destination)
    if sub_dirs is None:
        sub_dirs = ["test", "train"]
    for dir_name in sub_dirs:
        absolute_path = os.path.join(destination, dir_name)
        try:
            os.mkdir(absolute_path)
        except FileExistsError:
            shutil.rmtree(absolute_path)
            os.mkdir(absolute_path)


def degrade_images(destination):
    """Apply stochastic degradation pipeline to the images in a given dir"""
    train_images = glob(os.path.join(destination, "train_output") + "/*.*")
    test_images = glob(os.path.join(destination, "test_output") + "/*.*")

    for img_path in train_images:
        dst_path = os.path.join(destination, "train_input", os.path.basename(img_path))
        img = read_image(img_path)
        degraded_img = randomly_degrade(img) * 0.5 + 0.5
        save_image(degraded_img, dst_path)

    for img_path in test_images:
        dst_path = os.path.join(destination, "test_input", os.path.basename(img_path))
        img = read_image(img_path)
        degraded_img = randomly_degrade(img) * 0.5 + 0.5
        save_image(degraded_img, dst_path)


def split_images(
    source, destination, train_size=None, test_size=None, paired_dataset=False
):
    """Split images between train and test samples and move them to their corresponding dirs"""
    assert train_size is not None or test_size is not None, "Unspecified split fraction"
    train_size = train_size if train_size is not None else 1 - test_size
    images = sorted(glob(source + "/*.*"))
    train_images, test_images = train_test_split(images, train_size=train_size)

    sub_dirs = (
        ["train", "test"]
        if not paired_dataset
        else ["test_input", "test_output", "train_input", "train_output"]
    )
    make_dirs(destination, sub_dirs)

    for img_path in train_images:
        dst_path = os.path.join(
            destination,
            "train" if not paired_dataset else "train_output",
            os.path.basename(img_path),
        )
        shutil.copyfile(img_path, dst_path)

    for img_path in test_images:
        dst_path = os.path.join(
            destination,
            "test" if not paired_dataset else "test_output",
            os.path.basename(img_path),
        )
        shutil.copyfile(img_path, dst_path)


def enumerate_images(path, random=False):
    """Assign either numbers, or random symbols as names to images"""
    # Anyway assign random names first to avoid the issue of
    # erasing already existing files, e.g., 5.jpg
    for image_name in os.listdir(path):
        full_path = os.path.join(path, image_name)
        extension = "." + image_name.rpartition(".")[2]
        new_name = base64.b64encode(os.urandom(32)).decode()
        new_name = new_name.replace("/", "_") + extension
        new_path = os.path.join(path, new_name)
        os.rename(full_path, new_path)

    if not random:
        for i, image_name in enumerate(os.listdir(path), 1):
            full_path = os.path.join(path, image_name)
            extension = "." + image_name.rpartition(".")[2]
            new_name = f"{i}" + extension
            new_path = os.path.join(path, new_name)
            os.rename(full_path, new_path)


def sieve_images(path, k=20):
    """Keep only each k-th image, and remove the others"""
    for i, image_name in enumerate(os.listdir(path), 1):
        if i % k:
            image_path = os.path.join(path, image_name)
            os.remove(image_path)


def crop_images(source, destination, width, height):
    """Divide images into equal patches of specified size and save them. Partial crops are discarded"""
    image_files = [f for f in os.listdir(source)]

    for image_file in image_files:
        image_path = os.path.join(source, image_file)
        image_extension = image_file.rpartition(".")[2]
        image_name = image_file.rpartition(".")[0]
        image_tensor = read_image(image_path)

        crops = split_image(image_tensor, h0=height, w0=width, full_patches_only=True)

        for i, row in enumerate(crops, 1):
            for j, img in enumerate(row, 1):
                tensor = torch.from_numpy(img) * 0.5 + 0.5
                output_path = os.path.join(
                    destination, f"{image_name}_{i}-{j}.{image_extension}"
                )
                save_image(tensor, output_path)
