"""
Generate samples from a trained DDPM checkpoint.

Loads a trained model and runs the full reverse diffusion process
to generate images. Supports saving grids and individual images
for downstream evaluation (FID / IS).
"""

import os
import argparse
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from config import TrainConfig
from unet import UNet
from ddpm import DDPM
from dataset import unnormalize
from utils import save_samples_grid, load_checkpoint, set_seed


def generate_samples(config: TrainConfig, checkpoint_path: str, num_samples: int = 1000, save_individual: bool = True):
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)

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
    load_checkpoint(checkpoint_path, unet)
    unet.eval()

    output_dir = os.path.join(config.output_dir, "generated", config.noise_schedule)
    os.makedirs(output_dir, exist_ok=True)

    if save_individual:
        individual_dir = os.path.join(output_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)

    all_samples = []
    num_batches = (num_samples + config.sample_batch_size - 1) // config.sample_batch_size
    sample_count = 0

    print(f"Generating {num_samples} samples with {config.noise_schedule} schedule...")
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        current_batch = min(config.sample_batch_size, num_samples - sample_count)
        with torch.no_grad():
            samples = ddpm.sample((current_batch, config.channels, config.image_size, config.image_size))

        samples_unnorm = unnormalize(samples).clamp(0, 1)
        all_samples.append(samples_unnorm.cpu())

        if save_individual:
            for i in range(current_batch):
                save_image(
                    samples_unnorm[i],
                    os.path.join(individual_dir, f"sample_{sample_count + i:05d}.png"),
                )
        sample_count += current_batch

    all_samples = torch.cat(all_samples, dim=0)[:num_samples]

    grid_path = os.path.join(output_dir, "sample_grid.png")
    save_samples_grid(all_samples[:100] * 2 - 1, grid_path, nrow=10)
    print(f"Saved sample grid to {grid_path}")

    if save_individual:
        print(f"Saved {num_samples} individual samples to {os.path.join(output_dir, 'individual')}")

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained DDPM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        noise_schedule=args.schedule,
        num_timesteps=args.timesteps,
        device=args.device,
        output_dir=args.output_dir,
        sample_batch_size=args.batch_size,
        seed=args.seed,
    )

    generate_samples(config, args.checkpoint, args.num_samples)


if __name__ == "__main__":
    main()
