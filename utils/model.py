import os
from dataclasses import dataclass, field
from glob import glob

import torch
from skimage.metrics import structural_similarity as ssim
from torcheval.metrics import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from .config import TestPerformanceReport


def load_models(dir, device, config):
    """Restore weights from the last saved epoch in a specified dir, or return a new model if such do not exist"""

    def sort_by(ckpt_name):
        """Determine epoch  by checkpoint name"""
        last_underscore_idx = ckpt_name.rfind("_")
        return int(ckpt_name[last_underscore_idx + 2 :].partition(".")[0])

    generator_paths = sorted(
        glob(os.path.join(dir, f"{config.dataset_name}_generator") + "*.*"),
        key=sort_by,
    )
    generator = config.generator()
    if generator_paths:
        generator_loader = torch.load(generator_paths[-1], weights_only=False)
        generator.load_state_dict(generator_loader["model_state_dict"])
        generator.epochs_passed = generator_loader["epoch"]

    discriminator_paths = sorted(
        glob(os.path.join(dir, f"{config.dataset_name}_discriminator") + "*.*"),
        key=sort_by,
    )
    discriminator = config.discriminator()
    if discriminator_paths:
        discriminator_loader = torch.load(discriminator_paths[-1], weights_only=False)
        discriminator.load_state_dict(discriminator_loader["model_state_dict"])
        discriminator.epochs_passed = discriminator_loader["epoch"]

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    config.generator_opt = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    config.discriminator_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    return generator, discriminator


def save_model(model, optimizer, name, config):
    # rename .pt files
    dir = os.path.join(config.root, "weights")
    if not os.path.exists(dir):
        os.mkdir(dir)

    old_weights = glob(os.path.join(dir, f"{config.dataset_name}_{name}") + "*.*")
    for file in old_weights:
        os.remove(file)

    torch.save(
        {
            "epoch": model.epochs_passed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(dir, f"{config.dataset_name}_{name}_E{model.epochs_passed}.pt"),
    )


def evaluate(generator, test_dl, device, all_metrics=False):
    test_report = TestPerformanceReport()
    gen_batches, real_batches = [], []

    # Collect generated and real images
    for X_batch, Y_batch in test_dl:
        real_batches.append(Y_batch * 0.5 + 0.5)  # Rescale to [0, 1]
        X_batch = X_batch.to(device)
        with torch.set_grad_enabled(False):
            gen_batch = (
                generator(X_batch).detach().cpu() * 0.5 + 0.5
            )  # Rescale to [0, 1]
            gen_batches.append(gen_batch)

    # Concatenate all generated and real images
    fake_images = torch.cat(gen_batches, dim=0)
    real_images = torch.cat(real_batches, dim=0)

    # Calculate FID
    FID = FrechetInceptionDistance(device=device)
    FID.update(fake_images, False)
    FID.update(real_images, True)
    test_report.fid = FID.compute().detach().cpu()

    if all_metrics:
        # Calculate Inception Score (IS)
        IS = InceptionScore()
        IS.update(fake_images)
        test_report.inception_score = IS.compute().detach().cpu().item()

        # Calculate Kernel Inception Distance (KID)
        KID = KernelInceptionDistance(subset_size=100)
        KID.update(fake_images, real=False)
        KID.update(real_images, real=True)
        test_report.kid = (
            KID.compute().mean().detach().cpu().item()
        )  # KID returns a tensor with multiple values

    return test_report
