"""
LPIPS (Learned Perceptual Image Patch Similarity) implementation.

Based on: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
"""

import torch
import lpips as lpips_lib
from typing import Optional


class LPIPS:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric.

    Measures perceptual similarity between images.
    Lower is better (0 = identical, 1 = very different).

    Args:
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device to use

    Example:
        >>> lpips_metric = LPIPS(device='cuda')
        >>> distance = lpips_metric.compute(image1, image2)
        >>> print(f"LPIPS distance: {distance:.4f}")
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load LPIPS model
        self.model = lpips_lib.LPIPS(net=net).to(self.device)
        self.model.eval()

    def compute(
        self, images1: torch.Tensor, images2: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute LPIPS distance between images.

        Args:
            images1: First images (B, 3, H, W) in [0, 1] or [-1, 1]
            images2: Second images (B, 3, H, W) in [0, 1] or [-1, 1]
            normalize: If True, assume images in [0, 1] and normalize to [-1, 1]

        Returns:
            LPIPS distances (B,)
        """
        # Normalize to [-1, 1] if needed
        if normalize:
            images1 = images1 * 2 - 1
            images2 = images2 * 2 - 1

        # Move to device
        images1 = images1.to(self.device)
        images2 = images2.to(self.device)

        # Compute distance
        with torch.no_grad():
            distance = self.model(images1, images2)

        return distance.squeeze()

    def compute_mean(
        self, images1: torch.Tensor, images2: torch.Tensor, batch_size: Optional[int] = None
    ) -> float:
        """
        Compute mean LPIPS distance.

        Args:
            images1: First images (N, 3, H, W)
            images2: Second images (N, 3, H, W)
            batch_size: Process in batches

        Returns:
            Mean LPIPS distance
        """
        if batch_size is None:
            distances = self.compute(images1, images2)
            return float(distances.mean())

        # Process in batches
        total_distance = 0.0
        n_samples = 0

        for i in range(0, len(images1), batch_size):
            end = min(i + batch_size, len(images1))
            batch1 = images1[i:end]
            batch2 = images2[i:end]

            distances = self.compute(batch1, batch2)
            total_distance += distances.sum().item()
            n_samples += len(distances)

        return total_distance / n_samples
