"""
Noise schedule ablation study.

Trains DDPM with three noise schedules (linear, cosine, sigmoid) across
500 timesteps, logs training loss curves, and generates 1,000 samples
per configuration for FID / IS evaluation.

This script orchestrates the full ablation pipeline:
  1. Train each schedule configuration
  2. Generate samples from each trained model
  3. Evaluate FID and IS for each configuration
  4. Produce comparison plots
"""

import os
import json
import argparse
import torch

from config import TrainConfig
from train import train
from sample import generate_samples
from evaluate import compute_inception_score, evaluate_from_directory, HAS_TORCH_FIDELITY
from utils import (
    plot_loss_curves,
    plot_schedule_comparison,
    plot_snr_comparison,
    set_seed,
    load_checkpoint,
)
from dataset import unnormalize


SCHEDULES = ["linear", "cosine", "sigmoid"]


def run_ablation(base_config: TrainConfig):
    """Run the full ablation study across all noise schedules."""
    results = {}
    all_losses = {}

    ablation_dir = os.path.join(base_config.output_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # Plot schedule comparison before training
    plot_schedule_comparison(
        base_config.num_timesteps,
        os.path.join(ablation_dir, "schedule_comparison.png"),
        cosine_s=base_config.cosine_s,
    )
    plot_snr_comparison(
        base_config.num_timesteps,
        os.path.join(ablation_dir, "snr_comparison.png"),
        cosine_s=base_config.cosine_s,
    )
    print("Saved schedule comparison plots.")

    for schedule in SCHEDULES:
        print(f"\n{'='*60}")
        print(f"  Training with {schedule.upper()} schedule")
        print(f"{'='*60}\n")

        config = TrainConfig(
            noise_schedule=schedule,
            num_epochs=base_config.num_epochs,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            num_timesteps=base_config.num_timesteps,
            device=base_config.device,
            output_dir=base_config.output_dir,
            checkpoint_dir=base_config.checkpoint_dir,
            data_dir=base_config.data_dir,
            seed=base_config.seed,
        )

        # Train
        losses = train(config)
        all_losses[schedule] = losses

        # Find the latest checkpoint
        ckpt_dir = os.path.join(config.checkpoint_dir, schedule)
        checkpoints = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        if not checkpoints:
            print(f"No checkpoints found for {schedule}, skipping evaluation.")
            continue
        latest_ckpt = os.path.join(ckpt_dir, checkpoints[-1])

        # Generate 1000 samples
        print(f"\nGenerating 1000 samples for {schedule} schedule...")
        generate_samples(config, latest_ckpt, num_samples=1000, save_individual=True)

        # Evaluate
        generated_dir = os.path.join(config.output_dir, "generated", schedule, "individual")
        eval_results = {}

        if HAS_TORCH_FIDELITY and os.path.exists(generated_dir):
            eval_results = evaluate_from_directory(
                generated_dir,
                os.path.join(ablation_dir, f"{schedule}_metrics.json"),
            )
        else:
            print(f"  Skipping FID evaluation for {schedule} (torch-fidelity not available or no samples).")

        results[schedule] = {
            "final_loss": losses[-1] if losses else None,
            "avg_last_epoch_loss": sum(losses[-100:]) / min(len(losses), 100) if losses else None,
            **eval_results,
        }

    # Plot loss comparison
    plot_loss_curves(all_losses, os.path.join(ablation_dir, "loss_comparison.png"))
    print("\nSaved loss comparison plot.")

    # Save aggregated results
    results_path = os.path.join(ablation_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {results_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("  ABLATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Schedule':<12} {'Final Loss':<14} {'FID':<10} {'IS':<10}")
    print("-" * 46)
    for schedule, r in results.items():
        final_loss = f"{r.get('avg_last_epoch_loss', 'N/A'):.4f}" if r.get("avg_last_epoch_loss") else "N/A"
        fid = f"{r.get('fid', 'N/A'):.1f}" if r.get("fid") else "N/A"
        is_val = f"{r.get('is_mean', 'N/A'):.2f}" if r.get("is_mean") else "N/A"
        print(f"{schedule:<12} {final_loss:<14} {fid:<10} {is_val:<10}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run noise schedule ablation study")
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
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_timesteps=args.timesteps,
        device=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        seed=args.seed,
    )

    run_ablation(config)


if __name__ == "__main__":
    main()
