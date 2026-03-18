"""
Evaluation metrics for generated samples: FID and Inception Score (IS).

Uses torch-fidelity for robust FID/IS computation against CIFAR-10.
Also includes a standalone IS implementation for reference.
"""

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from tqdm import tqdm

HAS_TORCH_FIDELITY = True
try:
    import torch_fidelity
except ImportError:
    HAS_TORCH_FIDELITY = False


def compute_inception_score(samples: torch.Tensor, batch_size: int = 64, splits: int = 10) -> tuple:
    """
    Compute Inception Score using a pretrained Inception v3 model.

    Args:
        samples: Tensor of shape (N, 3, H, W) in [0, 1].
        batch_size: Batch size for inference.
        splits: Number of splits for IS computation.

    Returns:
        (mean IS, std IS) across splits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval()
    inception.to(device)

    resize = transforms.Resize((299, 299), antialias=True)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    classifier = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    classifier.eval()
    classifier.to(device)

    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Computing IS"):
            batch = resize(batch)
            batch = normalize(batch).to(device)
            logits = classifier(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.nn.functional.softmax(all_logits, dim=1).numpy()

    split_scores = []
    n = len(probs)
    for k in range(splits):
        part = probs[k * (n // splits): (k + 1) * (n // splits)]
        py = np.mean(part, axis=0)
        scores = []
        for p in part:
            kl = np.sum(p * (np.log(p + 1e-10) - np.log(py + 1e-10)))
            scores.append(kl)
        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))


def compute_fid_torch_fidelity(generated_dir: str, dataset_name: str = "cifar10-train") -> float:
    """Compute FID using torch-fidelity against real CIFAR-10 stats."""
    if not HAS_TORCH_FIDELITY:
        raise ImportError("torch-fidelity is required. Install with: pip install torch-fidelity")

    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=dataset_name,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
    )
    return metrics


def evaluate_from_directory(generated_dir: str, output_path: str = None):
    """Run full evaluation on a directory of generated images."""
    print(f"Evaluating samples in: {generated_dir}")

    results = {}
    if HAS_TORCH_FIDELITY:
        print("Computing FID and IS via torch-fidelity...")
        metrics = compute_fid_torch_fidelity(generated_dir)
        results["fid"] = metrics.get("frechet_inception_distance", None)
        results["is_mean"] = metrics.get("inception_score_mean", None)
        results["is_std"] = metrics.get("inception_score_std", None)
        print(f"  FID: {results['fid']:.2f}")
        print(f"  IS:  {results['is_mean']:.2f} ± {results['is_std']:.2f}")
    else:
        print("torch-fidelity not available, using standalone IS computation...")
        from torchvision.datasets import ImageFolder
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ImageFolder(os.path.dirname(generated_dir), transform=transform)
        samples = torch.stack([img for img, _ in dataset])
        is_mean, is_std = compute_inception_score(samples)
        results["is_mean"] = is_mean
        results["is_std"] = is_std
        print(f"  IS: {is_mean:.2f} ± {is_std:.2f}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated samples")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory with generated PNG images")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    evaluate_from_directory(args.generated_dir, args.output)


if __name__ == "__main__":
    main()
