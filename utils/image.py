import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import crop


def visualize(
    tensors,
    n_col=None,
    max_rows=np.inf,
    row_captions=None,
    save_to=None,
    height=5,
    width=5,
    title="",
):
    if n_col is None:
        n_col = min(len(tensors) // 2, 3)

    n_row = len(tensors) // n_col
    assert n_row * n_col == tensors.shape[0]
    if row_captions:
        assert len(row_captions) == n_row

    # Set the figure size
    plt.figure(figsize=(n_col * width, n_row * height))
    plt.title(title, fontsize=10)
    plt.axis("off")

    for i in range(n_row):
        if i + 1 > max_rows:
            break
        for j in range(n_col):
            idx = j + i * n_col
            plt.subplot(n_row, n_col, idx + 1)
            plt.imshow(tensors[idx].permute(1, 2, 0) * 0.5 + 0.5)
            if row_captions:
                plt.title(row_captions[i])
            plt.axis("off")

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")

    plt.show()


def read_image(path=None):
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    if path is not None:
        img = Image.open(path)
    img = transforms(img)
    return img


def split_image(input_tensor, h0, w0, full_patches_only=False):
    """Set full_pacthes_only to True if want to include partial patches of the image"""
    patches = []
    orig_height, orig_width = input_tensor.shape[-2:]
    round = math.floor if full_patches_only else math.ceil
    n_rows = round(orig_height / h0)
    n_cols = round(orig_width / w0)
    for i in range(n_rows):
        cur_row = []
        for j in range(n_cols):
            top, left = i * h0, j * w0
            patch = crop(img=input_tensor, top=top, left=left, height=h0, width=w0)
            cur_row.append(patch)
        patches.append(cur_row)

    return np.array(patches)


def restore_image(patches, orig_height, orig_width):
    """Construct an image back from patches to meet original height & width"""
    n_rows, n_cols = patches.shape[:2]
    rows = []
    for i in range(n_rows):
        cur_row = []
        for j in range(n_cols):
            patch = patches[i, j]
            cur_row.append(patch)
        cur_row = np.concatenate(cur_row, axis=-1)
        rows.append(cur_row)

    img = np.concatenate(rows, axis=-2)
    tensor = torch.from_numpy(img)

    top, left = 0, 0
    restored_img = crop(
        img=tensor, top=top, left=left, height=orig_height, width=orig_width
    )
    return restored_img * 0.5 + 0.5
