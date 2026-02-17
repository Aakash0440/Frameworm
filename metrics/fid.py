"""
FID (Fréchet Inception Distance) implementation.

Based on: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision import models
from tqdm import tqdm


class InceptionV3Features(nn.Module):
    """
    InceptionV3 feature extractor for FID.

    Extracts features from the final average pooling layer.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained Inception V3
        inception = models.inception_v3(pretrained=True, transform_input=False)

        # Remove final classification layer
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features.

        Args:
            x: Images (B, 3, 299, 299) in range [0, 1]

        Returns:
            Features (B, 2048)
        """
        # Normalize to [-1, 1] (Inception V3 expects this)
        x = 2 * x - 1

        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)

        return features


def calculate_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6
) -> float:
    """
    Calculate Fréchet distance between two Gaussian distributions.

    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small value for numerical stability

    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Means have different shapes"
    assert sigma1.shape == sigma2.shape, "Covariances have different shapes"

    # Calculate difference of means
    diff = mu1 - mu2

    # Calculate sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical issues
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    # Calculate Fréchet distance
    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return float(fid)


class FID:
    """
    Fréchet Inception Distance (FID) metric.

    Measures the quality and diversity of generated images.
    Lower is better (0 = identical distributions).

    Args:
        device: Device to use for computation
        batch_size: Batch size for feature extraction

    Example:
        >>> fid = FID(device='cuda')
        >>> score = fid.compute(real_images, generated_images)
        >>> print(f"FID: {score:.2f}")
    """

    def __init__(self, device: str = "cuda", batch_size: int = 50):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Load Inception V3
        self.inception = InceptionV3Features().to(self.device)

    def extract_features(self, images: torch.Tensor, show_progress: bool = True) -> np.ndarray:
        """
        Extract Inception features from images.

        Args:
            images: Images tensor (N, 3, H, W) in range [0, 1]
            show_progress: Show progress bar

        Returns:
            Features (N, 2048)
        """
        self.inception.eval()

        features_list = []

        # Resize images to 299x299 (Inception V3 input size)
        if images.size(2) != 299 or images.size(3) != 299:
            images = torch.nn.functional.interpolate(
                images, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Extract features in batches
        n_batches = (len(images) + self.batch_size - 1) // self.batch_size

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")

        with torch.no_grad():
            for i in iterator:
                start = i * self.batch_size
                end = min(start + self.batch_size, len(images))

                batch = images[start:end].to(self.device)
                features = self.inception(batch)
                features_list.append(features.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        return features

    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and covariance of features.

        Args:
            features: Feature array (N, D)

        Returns:
            (mean, covariance)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def compute(
        self, real_images: torch.Tensor, generated_images: torch.Tensor, show_progress: bool = True
    ) -> float:
        """
        Compute FID between real and generated images.

        Args:
            real_images: Real images (N, 3, H, W) in [0, 1]
            generated_images: Generated images (M, 3, H, W) in [0, 1]
            show_progress: Show progress bars

        Returns:
            FID score (lower is better)
        """
        # Extract features
        if show_progress:
            print("Extracting features from real images...")
        real_features = self.extract_features(real_images, show_progress)

        if show_progress:
            print("Extracting features from generated images...")
        gen_features = self.extract_features(generated_images, show_progress)

        # Calculate statistics
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)

        # Calculate FID
        fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        return fid_score

    def compute_from_loader(
        self, real_loader, generated_loader, max_samples: Optional[int] = None
    ) -> float:
        """
        Compute FID from data loaders.

        Args:
            real_loader: DataLoader for real images
            generated_loader: DataLoader for generated images
            max_samples: Maximum number of samples to use

        Returns:
            FID score
        """
        # Collect images
        real_images = []
        gen_images = []

        print("Loading real images...")
        for i, batch in enumerate(tqdm(real_loader)):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            real_images.append(batch)

            if max_samples and len(real_images) * batch.size(0) >= max_samples:
                break

        print("Loading generated images...")
        for i, batch in enumerate(tqdm(generated_loader)):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            gen_images.append(batch)

            if max_samples and len(gen_images) * batch.size(0) >= max_samples:
                break

        real_images = torch.cat(real_images, dim=0)
        gen_images = torch.cat(gen_images, dim=0)

        if max_samples:
            real_images = real_images[:max_samples]
            gen_images = gen_images[:max_samples]

        # Compute FID
        return self.compute(real_images, gen_images)
