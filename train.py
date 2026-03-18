"""
Training script for DDPM on CIFAR-10.

Trains a U-Net score estimator using the simplified DDPM objective:
    L = E_{t, x_0, eps} [|| eps - eps_theta(x_t, t) ||^2]

Supports linear, cosine, and sigmoid noise schedules.
Logs training metrics to TensorBoard.
"""

import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TrainConfig
from unet import UNet
from ddpm import DDPM
from dataset import get_cifar10_dataloader
from utils import EMA, save_checkpoint, save_samples_grid, set_seed


def train(config: TrainConfig):
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    print(f"Using device: {device}")
    print(f"Noise schedule: {config.noise_schedule}")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    dataloader = get_cifar10_dataloader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
    )

    unet = UNet(
        in_channels=config.channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        time_emb_dim=config.time_emb_dim,
        image_size=config.image_size,
    ).to(device)

    ddpm = DDPM(unet, config).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    ema = EMA(unet, decay=config.ema_decay)

    param_count = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    log_dir = os.path.join(config.output_dir, "logs", config.noise_schedule)
    writer = SummaryWriter(log_dir)

    global_step = 0
    losses = []

    for epoch in range(1, config.num_epochs + 1):
        unet.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            loss = ddpm.training_loss(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            ema.update(unet)

            epoch_loss += loss.item()
            global_step += 1
            losses.append(loss.item())

            if global_step % config.log_interval == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        if epoch % config.sample_interval == 0:
            unet.eval()
            ema.apply(unet)
            with torch.no_grad():
                samples = ddpm.sample((64, config.channels, config.image_size, config.image_size))
            save_path = os.path.join(config.output_dir, "samples", config.noise_schedule, f"epoch_{epoch:04d}.png")
            save_samples_grid(samples, save_path, nrow=8)
            writer.add_images("samples", (samples[:16].clamp(-1, 1) + 1) / 2, epoch)
            print(f"  Saved samples to {save_path}")

        if epoch % config.save_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, config.noise_schedule, f"ddpm_epoch_{epoch:04d}.pt")
            save_checkpoint(unet, optimizer, epoch, avg_loss, config, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

    writer.close()
    return losses


def main():
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        noise_schedule=args.schedule,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_timesteps=args.timesteps,
        device=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        seed=args.seed,
    )

    train(config)


if __name__ == "__main__":
    main()
