"""
Example: Comprehensive Search Method Comparison

Compares Grid, Random, and Bayesian search on same problem.
"""

from core import Config, get_model
from training import Trainer
from search import GridSearch, RandomSearch, SearchAnalyzer
from search.space import Real, Integer
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

try:
    from search import BayesianSearch

    BAYESIAN_AVAILABLE = True
except:
    BAYESIAN_AVAILABLE = False


def get_mnist_loaders(batch_size=128):
    """Get MNIST data loaders (subset for fast search)"""
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST("data", train=False, transform=transform)

    # Use subsets for faster search
    train_subset = torch.utils.data.Subset(train_dataset, range(3000))
    val_subset = torch.utils.data.Subset(val_dataset, range(500))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    return train_loader, val_loader


def train_vae_quick(config: Config) -> dict:
    """Quick VAE training for search"""
    train_loader, val_loader = get_mnist_loaders(config.training.batch_size)

    vae = get_model("vae")(config)
    optimizer = torch.optim.Adam(vae.parameters(), lr=config.training.lr)

    trainer = Trainer(
        model=vae, optimizer=optimizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Quick training (3 epochs for search)
    trainer.train(train_loader, val_loader, epochs=3)

    val_loss = trainer.state.val_metrics["loss"][-1]
    train_loss = trainer.state.train_metrics["loss"][-1]

    return {"val_loss": val_loss, "train_loss": train_loss}


def compare_search_methods():
    """Compare different search methods"""
    print("=" * 60)
    print("SEARCH METHOD COMPARISON")
    print("=" * 60)

    # Base configuration
    base_config = Config("configs/models/vae/vanilla.yaml")
    base_config.training.epochs = 3  # Quick for search
    base_config.training.batch_size = 128
    base_config.training.lr = 0.001

    # Search space
    search_space_grid = {
        "training.lr": [0.001, 0.0005, 0.0001],
        "training.batch_size": [64, 128, 256],
    }

    search_space_continuous = {
        "training.lr": Real(1e-4, 1e-2, log=True),
        "training.batch_size": Integer(32, 256, log=True),
    }

    results = {}

    # 1. Grid Search
    print("\n" + "=" * 60)
    print("1. GRID SEARCH")
    print("=" * 60)

    start_time = time.time()

    grid_search = GridSearch(
        base_config=base_config, search_space=search_space_grid, metric="val_loss", mode="min"
    )

    best_config, best_score = grid_search.run(
        train_fn=train_vae_quick, experiment_root="experiments/comparison/grid", verbose=True
    )

    grid_time = time.time() - start_time

    results["Grid Search"] = {
        "best_score": best_score,
        "best_config": best_config,
        "time": grid_time,
        "n_trials": len(grid_search.results),
        "results": grid_search.results,
    }

    print(
        f"\nGrid Search: {best_score:.4f} in {grid_time:.1f}s ({len(grid_search.results)} trials)"
    )

    # 2. Random Search
    print("\n" + "=" * 60)
    print("2. RANDOM SEARCH")
    print("=" * 60)

    start_time = time.time()

    random_search = RandomSearch(
        base_config=base_config,
        search_space=search_space_continuous,
        metric="val_loss",
        mode="min",
        n_trials=15,  # Same budget as grid
        random_state=42,
    )

    best_config, best_score = random_search.run(
        train_fn=train_vae_quick, experiment_root="experiments/comparison/random", verbose=True
    )

    random_time = time.time() - start_time

    results["Random Search"] = {
        "best_score": best_score,
        "best_config": best_config,
        "time": random_time,
        "n_trials": len(random_search.results),
        "results": random_search.results,
    }

    print(
        f"\nRandom Search: {best_score:.4f} in {random_time:.1f}s ({len(random_search.results)} trials)"
    )

    # 3. Bayesian Optimization
    if BAYESIAN_AVAILABLE:
        print("\n" + "=" * 60)
        print("3. BAYESIAN OPTIMIZATION")
        print("=" * 60)

        start_time = time.time()

        bayes_search = BayesianSearch(
            base_config=base_config,
            search_space=search_space_continuous,
            metric="val_loss",
            mode="min",
            n_trials=15,
            n_initial_points=5,
            random_state=42,
        )

        best_config, best_score = bayes_search.run(
            train_fn=train_vae_quick,
            experiment_root="experiments/comparison/bayesian",
            verbose=True,
        )

        bayes_time = time.time() - start_time

        results["Bayesian Optimization"] = {
            "best_score": best_score,
            "best_config": best_config,
            "time": bayes_time,
            "n_trials": len(bayes_search.results),
            "results": bayes_search.results,
        }

        print(
            f"\nBayesian Optimization: {best_score:.4f} in {bayes_time:.1f}s ({len(bayes_search.results)} trials)"
        )
    else:
        print("\n⚠️  Bayesian Optimization not available (install scikit-optimize)")

    # Comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    print(f"\n{'Method':<25} {'Best Score':<12} {'Time (s)':<10} {'Trials':<8}")
    print("-" * 60)

    for method, data in results.items():
        print(
            f"{method:<25} {data['best_score']:<12.4f} {data['time']:<10.1f} {data['n_trials']:<8}"
        )

    # Find winner
    best_method = min(results.items(), key=lambda x: x[1]["best_score"])
    print(f"\n✓ Winner: {best_method[0]} with score {best_method[1]['best_score']:.4f}")

    # Plot comparison
    plot_comparison(results)

    return results


def plot_comparison(results: dict):
    """Plot search method comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Best scores
    ax = axes[0]
    methods = list(results.keys())
    scores = [results[m]["best_score"] for m in methods]

    ax.bar(methods, scores)
    ax.set_ylabel("Best Val Loss")
    ax.set_title("Best Score Comparison")
    ax.tick_params(axis="x", rotation=45)

    # Plot 2: Convergence
    ax = axes[1]
    for method, data in results.items():
        scores = [r["score"] for r in data["results"]]
        cummin = []
        current_min = float("inf")
        for s in scores:
            current_min = min(current_min, s)
            cummin.append(current_min)
        ax.plot(cummin, label=method, marker="o", markersize=3)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Score So Far")
    ax.set_title("Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Efficiency (score vs time)
    ax = axes[2]
    for method, data in results.items():
        ax.scatter(data["time"], data["best_score"], s=100, label=method)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Best Score")
    ax.set_title("Efficiency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("search_comparison.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved comparison plot to search_comparison.png")


def main():
    print("Search Method Comparison Example")
    print("=" * 60)

    results = compare_search_methods()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
