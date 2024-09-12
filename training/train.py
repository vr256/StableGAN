import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output
from torchvision.utils import save_image
from utils.image import visualize
from utils.model import evaluate, save_model


def train_discriminator(generator, discriminator, X_batch, Y_batch, config):

    config.discriminator_opt.zero_grad()

    Y_gen = generator(X_batch)
    prob_gen = discriminator(torch.cat([X_batch, Y_gen], dim=1))
    prob_real = discriminator(torch.cat([X_batch, Y_batch], dim=1))

    # ! Should I plot discriminator accuracy ?
    # labels_real = torch.ones_like(prob_real)
    # labels_gen = torch.zeros_like(prob_gen)
    # correct_real = (prob_real >= 0.5).float().eq(labels_real).sum().item()
    # correct_gen = (prob_gen < 0.5).float().eq(labels_gen).sum().item()
    current_accuracy = 0.5

    loss = config.discriminator_loss(prob_real, prob_gen)
    current_loss = loss.item()

    loss.backward()
    config.discriminator_opt.step()

    return current_loss, current_accuracy


def train_generator(generator, discriminator, X_batch, Y_batch, config):
    config.generator_opt.zero_grad()

    Y_gen = generator(X_batch)
    prob_gen = discriminator(torch.cat([X_batch, Y_gen], dim=1))

    loss = config.generator_loss(Y_gen, prob_gen, Y_batch)
    current_loss = loss.item()

    loss.backward()
    config.generator_opt.step()

    return current_loss


def train(
    generator,
    discriminator,
    epochs,
    train_dl,
    test_dl,
    device,
    config,
    name,
    extra_dl=None,
    nu=5,
    epochs_to_save=20,
):
    if torch.cuda.is_available():
        device_name = f"({torch.cuda.get_device_name(device)})"
    else:
        device_name = ""
    training_text = f"Training on {device} {device_name}\n"
    model_text = (
        f"Weights restored from a model trained on {generator.epochs_passed} epochs\n"
    )
    print(training_text)
    restored = generator.epochs_passed > 0
    if restored:
        print(model_text)

    X_test, Y_test = next(iter(test_dl))

    # Check if scores.csv exists and initialize columns if necessary
    scores_path = os.path.join(
        config.root, "scores", f"{config.dataset_name}_{name}.csv"
    )
    if not os.path.exists(scores_path):
        with open(scores_path, "w") as file:
            file.write("fid,disc_loss,gen_loss")
    else:
        df = pd.read_csv(scores_path)

    score = ""

    for epoch in range(1, epochs + 1):
        df = pd.read_csv(scores_path).tail(nu)
        total_disc_accuracy = df["disc_loss"].sum() if "disc_loss" in df.columns else 0
        total_gen_loss = df["gen_loss"].sum() if "gen_loss" in df.columns else 0

        adjusted_disc_accuracy = (
            df["disc_loss"].sum() / total_disc_accuracy
            if total_disc_accuracy != 0
            else 0
        )
        adjusted_gen_loss = (
            df["gen_loss"].sum() / total_gen_loss if total_gen_loss != 0 else 0
        )
        diff = adjusted_disc_accuracy - adjusted_gen_loss

        epoch_disc_accuracy = 0
        epoch_disc_loss = 0
        epoch_gen_loss = 0
        total_correct = 0
        total_samples = 0

        for X_batch, Y_batch in train_dl:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            batch_size = X_batch.size(0)
            total_samples += batch_size

            disc_loss, disc_accuracy = train_discriminator(
                generator=generator,
                discriminator=discriminator,
                X_batch=X_batch,
                Y_batch=Y_batch,
                config=config,
            )
            epoch_disc_accuracy += disc_accuracy * batch_size
            total_correct += disc_accuracy * batch_size

            gen_loss = train_generator(
                generator=generator,
                discriminator=discriminator,
                X_batch=X_batch,
                Y_batch=Y_batch,
                config=config,
            )
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss

        # Average accuracy over the total number of samples
        epoch_disc_accuracy /= total_samples

        # Evaluation on test dataset
        Y_gen_test = generator(X_test.to(device)).detach().cpu()
        test_report = evaluate(generator, test_dl, device)
        fid_score = test_report.fid

        score += f"\n{fid_score},{epoch_disc_loss},{epoch_gen_loss}"

        # Saving
        if epoch % epochs_to_save == 0:
            generator.epochs_passed += epochs_to_save
            discriminator.epochs_passed += epochs_to_save

            save_model(generator, config.generator_opt, "generator", config)
            save_model(discriminator, config.discriminator_opt, "discriminator", config)

            with open(scores_path, "a") as file:
                file.write(score)
                score = ""

            dir = os.path.join(config.root, "results")
            if not os.path.exists(dir):
                os.mkdir(dir)
            save_image(
                torch.cat((X_test[0], Y_gen_test[0], Y_test[0]), dim=2) * 0.5 + 0.5,
                os.path.join(
                    dir, f"{config.dataset_name}_{generator.epochs_passed}.jpg"
                ),
            )

        clear_output(wait=True)

        print(training_text)
        if restored > 0:
            print(model_text)
        print(f"Epoch {epoch}/{epochs}")

        # Visualization
        n_col = min(config.batch_size, 3)
        batch_size = config.batch_size
        indexes = np.random.randint(0, batch_size, n_col)
        images_to_show = torch.cat(
            [X_test[indexes], Y_gen_test[indexes], Y_test[indexes]], dim=0
        )
        visualize(
            images_to_show,
            n_col=n_col,
            row_captions=["Input", "Gen", "Real"],
            save_to="examples.jpg",
        )

        torch.cuda.empty_cache()
        gc.collect()
