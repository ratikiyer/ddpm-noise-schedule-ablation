from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    image_size: int = 32
    channels: int = 3
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 2e-4
    num_timesteps: int = 500
    noise_schedule: str = "cosine"  # "linear", "cosine", "sigmoid"

    # U-Net architecture
    base_channels: int = 128
    channel_mults: tuple = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16,)
    dropout: float = 0.1
    time_emb_dim: int = 512

    # Cosine schedule
    cosine_s: float = 0.008

    # Sigmoid schedule
    sigmoid_start: float = -3.0
    sigmoid_end: float = 3.0

    # Linear schedule
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Evaluation
    num_samples: int = 1000
    sample_batch_size: int = 64

    # Logging
    log_interval: int = 100
    save_interval: int = 10
    sample_interval: int = 5

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    # Device
    device: str = "cuda"
    num_workers: int = 4

    # EMA
    ema_decay: float = 0.9999

    seed: int = 42
