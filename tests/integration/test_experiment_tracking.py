import shutil
import tempfile
from pathlib import Path

import pytest

from experiment.experiment import Experiment


def test_experiment_tracking(tmp_path):
    """
    Integration test for Experiment tracking:
    - creation
    - metric logging
    - artifact logging
    - metric retrieval
    """

    root_dir = tmp_path / "test_experiments"

    # Create experiment
    exp = Experiment(
        name="test-experiment",
        config={'lr': 0.001, 'batch_size': 128},
        description="Test experiment",
        tags=["test", "baseline"],
        root_dir=str(root_dir)
    )

    with exp:
        # Simulate training
        for epoch in range(3):
            for step in range(10):
                exp.log_metric(
                    "loss",
                    1.0 / (step + 1),
                    step=epoch * 10 + step,
                    epoch=epoch
                )

            exp.log_metrics(
                {
                    'train_loss': 0.5,
                    'val_loss': 0.6
                },
                epoch=epoch,
                metric_type='val'
            )

        # Log artifact
        temp_file = tmp_path / "artifact.txt"
        temp_file.write_text("test artifact")

        exp.log_artifact(str(temp_file), artifact_type='test')

    # Assertions

    # Experiment completed
    assert exp.status in ["completed", "finished", "success"]

    # Metrics recorded correctly
    metrics = exp.get_metrics('loss')
    assert len(metrics) == 30  # 3 epochs * 10 steps

    # Artifact directory exists
    assert root_dir.exists()

    # Cleanup handled automatically by tmp_path
