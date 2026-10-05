import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DISC,
    CHECKPOINT_GEN,
    DEVICE,
    LEARNING_RATE,
    LOAD_MODEL,
    NUM_EPOCHS,
    NUM_WORKERS,
    SAVE_MODEL,
)
from dataset import MyImageFolder
from loss import VGGLoss
from model import Discriminator, Generator
from utils import load_checkpoint, plot_examples, save_checkpoint


torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(DEVICE)
        low_res = low_res.to(DEVICE)

        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if idx % 200 == 0:
            plot_examples("test_images/", gen)


def main():
    dataset_root = "new_data/" if os.path.exists("new_data/") else "images"
    if not os.path.exists(dataset_root):
        print(f"Training cancelled: '{dataset_root}' directory not found.")
        return

    dataset = MyImageFolder(root_dir=dataset_root)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=3).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if LOAD_MODEL:
        if os.path.exists(CHECKPOINT_GEN) and os.path.exists(CHECKPOINT_DISC):
            load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
            load_checkpoint(CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE)
        else:
            print("Checkpoints not found. Starting training from scratch.")

    epochs_to_run = min(1, NUM_EPOCHS)
    for epoch in range(epochs_to_run):
        print(f"Epoch [{epoch + 1}/{epochs_to_run}]")
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

        if SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)


if __name__ == "__main__":
    main()