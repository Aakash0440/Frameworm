"""
Inception Score (IS) implementation.

Based on: "Improved Techniques for Training GANs" (Salimans et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from typing import Tuple
from tqdm import tqdm


class InceptionV3Classifier(nn.Module):
    """
    InceptionV3 for classification (Inception Score).
    """

    def __init__(self):
        super().__init__()

        # Load pretrained Inception V3 with classification head
        self.inception = models.inception_v3(pretrained=True, transform_input=False)

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify images.

        Args:
            x: Images (B, 3, 299, 299) in range [0, 1]

        Returns:
            Class probabilities (B, 1000)
        """
        # Normalize to [-1, 1]
        x = 2 * x - 1

        # Get logits
        logits = self.inception(x)

        # Apply softmax
        probs = F.softmax(logits, dim=1)

        return probs


class InceptionScore:
    """
    Inception Score (IS) metric.

    Measures quality and diversity of generated images.
    Higher is better (typical range: 1-10+).

    Args:
        device: Device to use
        batch_size: Batch size for inference
        splits: Number of splits for computing std

    Example:
        >>> inception_score = InceptionScore(device='cuda')
        >>> score, std = inception_score.compute(generated_images)
        >>> print(f"IS: {score:.2f} ± {std:.2f}")
    """

    def __init__(self, device: str = "cuda", batch_size: int = 50, splits: int = 10):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.splits = splits

        # Load Inception V3
        self.inception = InceptionV3Classifier().to(self.device)

    def get_predictions(self, images: torch.Tensor, show_progress: bool = True) -> np.ndarray:
        """
        Get class predictions for images.

        Args:
            images: Images (N, 3, H, W) in [0, 1]
            show_progress: Show progress bar

        Returns:
            Predictions (N, 1000)
        """
        self.inception.eval()

        predictions_list = []

        # Resize to 299x299
        if images.size(2) != 299 or images.size(3) != 299:
            images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

        # Get predictions in batches
        n_batches = (len(images) + self.batch_size - 1) // self.batch_size

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing predictions")

        with torch.no_grad():
            for i in iterator:
                start = i * self.batch_size
                end = min(start + self.batch_size, len(images))

                batch = images[start:end].to(self.device)
                preds = self.inception(batch)
                predictions_list.append(preds.cpu().numpy())

        predictions = np.concatenate(predictions_list, axis=0)
        return predictions

    def calculate_inception_score(self, predictions: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Inception Score from predictions.

        Args:
            predictions: Class predictions (N, 1000)

        Returns:
            (mean_score, std_score)
        """
        # Split into parts
        split_size = predictions.shape[0] // self.splits
        scores = []

        for i in range(self.splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self.splits - 1 else predictions.shape[0]

            part = predictions[start:end]

            # Marginal distribution
            py = np.mean(part, axis=0)

            # KL divergence
            kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
            kl = np.mean(np.sum(kl, axis=1))

            # Inception Score
            scores.append(np.exp(kl))

        return float(np.mean(scores)), float(np.std(scores))

    def compute(self, images: torch.Tensor, show_progress: bool = True) -> Tuple[float, float]:
        """
        Compute Inception Score for images.

        Args:
            images: Generated images (N, 3, H, W) in [0, 1]
            show_progress: Show progress

        Returns:
            (mean_score, std_score)
        """
        # Get predictions
        predictions = self.get_predictions(images, show_progress)

        # Calculate IS
        score, std = self.calculate_inception_score(predictions)

        return score, std
