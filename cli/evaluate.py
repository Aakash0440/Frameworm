"""
Model evaluation command.
"""

import torch
from click import echo

from core import Config, get_model
from metrics import MetricEvaluator


def run_evaluation(config_path: str, checkpoint_path: str, metrics: list, num_samples: int = 10000):
    """Run model evaluation"""

    # Load config
    config = Config(config_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Create model
    model = get_model(config.model.type)(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    echo(f"✓ Model loaded from: {checkpoint_path}")

    # Create evaluator
    evaluator = MetricEvaluator(
        metrics=metrics,
        real_data=None,  # Placeholder
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    echo(f"Computing metrics: {metrics}")

    # Evaluate
    results = evaluator.evaluate(model, num_samples=num_samples)

    echo("\n" + "=" * 60)
    echo("EVALUATION RESULTS")
    echo("=" * 60)

    for metric, value in results.items():
        echo(f"{metric}: {value:.4f}")

    echo("=" * 60)
