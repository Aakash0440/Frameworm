"""Tests for metrics"""

import pytest
import torch

from metrics import FID, LPIPS, InceptionScore
from metrics.evaluator import MetricEvaluator


class TestFID:
    def test_fid_computation(self):
        fid = FID(device="cpu", batch_size=10)

        real = torch.rand(50, 3, 64, 64)
        fake = torch.rand(50, 3, 64, 64)

        score = fid.compute(real, fake, show_progress=False)

        assert isinstance(score, float)
        assert score >= 0

    def test_fid_identical(self):
        fid = FID(device="cpu", batch_size=10)

        images = torch.rand(50, 3, 64, 64)
        score = fid.compute(images, images, show_progress=False)

        assert score < 1.0  # Should be very small


class TestInceptionScore:
    def test_is_computation(self):
        inception_score = InceptionScore(device="cpu", batch_size=10, splits=2)

        images = torch.rand(50, 3, 64, 64)
        score, std = inception_score.compute(images, show_progress=False)

        assert isinstance(score, float)
        assert isinstance(std, float)
        assert score >= 1.0  # IS is always >= 1


class TestLPIPS:
    def test_lpips_computation(self):
        lpips = LPIPS(device="cpu")

        img1 = torch.rand(10, 3, 64, 64)
        img2 = torch.rand(10, 3, 64, 64)

        distances = lpips.compute(img1, img2)

        assert len(distances) == 10
        assert (distances >= 0).all()

    def test_lpips_identical(self):
        lpips = LPIPS(device="cpu")

        images = torch.rand(10, 3, 64, 64)
        distances = lpips.compute(images, images)

        assert (distances < 0.01).all()  # Should be very small


class TestMetricEvaluator:
    def test_evaluator_creation(self):
        real_images = torch.rand(100, 3, 64, 64)

        evaluator = MetricEvaluator(metrics=["fid", "is"], real_data=real_images, device="cpu")

        assert "fid" in evaluator.metric_objects
        assert "is" in evaluator.metric_objects
