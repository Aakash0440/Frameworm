"""
Example: Hyperparameter Search

Demonstrates:
- Grid search
- Random search
- Search analysis
- Integration with experiments
"""

from core import Config, get_model
from training import Trainer
from search import GridSearch, RandomSearch, SearchAnalyzer
from search.space import Real, Integer
from metrics import MetricEvaluator, FID
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=128):
    """Get MNIST data loaders"""
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Use subset for faster search
    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    val_subset = torch.utils.data.Subset(val_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_vae(config: Config) -> dict:
    """
    Training function for search.
    
    Args:
        config: Configuration to evaluate
        
    Returns:
        Dictionary with metrics
    """
    # Get data
    train_loader, val_loader = get_mnist_loaders(config.training.batch_size)
    
    # Create model
    vae = get_model("vae")(config)
    optimizer = torch.optim.Adam(vae.parameters(), lr=config.training.lr)
    
    # Train
    trainer = Trainer(
        model=vae,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Quick training (reduced epochs for search)
    trainer.train(train_loader, val_loader, epochs=config.training.epochs)
    
    # Get final validation loss
    val_loss = trainer.state.val_metrics['loss'][-1]
    train_loss = trainer.state.train_metrics['loss'][-1]
    
    return {
        'val_loss': val_loss,
        'train_loss': train_loss
    }


def example_grid_search():
    """Example: Grid search"""
    print("\n" + "="*60)
    print("GRID SEARCH EXAMPLE")
    print("="*60)
    
    # Base configuration
    base_config = Config('configs/models/vae/vanilla.yaml')
    base_config.training.epochs = 5  # Quick training for search
    base_config.training.batch_size = 128
    base_config.training.lr = 0.001
    
    # Define search space
    search_space = {
        'training.lr': [0.001, 0.0005, 0.0001],
        'training.batch_size': [64, 128, 256]
    }
    
    # Create search
    search = GridSearch(
        base_config=base_config,
        search_space=search_space,
        metric='val_loss',
        mode='min'
    )
    
    print(f"\nSearch space: {search_space}")
    print(f"Total configurations: {3 * 3} = 9")
    
    # Run search
    best_config, best_score = search.run(
        train_fn=train_vae,
        n_jobs=1,
        experiment_root='experiments/grid_search',
        verbose=True
    )
    
    # Analyze results
    analyzer = SearchAnalyzer(search.results)
    analyzer.print_summary()
    analyzer.plot_convergence(save_path='grid_search_convergence.png')
    
    # Save results
    search.save_results('grid_search_results.json')
    print(f"\n✓ Results saved to grid_search_results.json")
    
    return best_config, best_score


def example_random_search():
    """Example: Random search"""
    print("\n" + "="*60)
    print("RANDOM SEARCH EXAMPLE")
    print("="*60)
    
    # Base configuration
    base_config = Config('configs/models/vae/vanilla.yaml')
    base_config.training.epochs = 5
    base_config.training.batch_size = 128
    base_config.training.lr = 0.001
    
    # Define search space (continuous)
    search_space = {
        'training.lr': Real(1e-5, 1e-2, log=True),
        'training.batch_size': Integer(32, 256, log=True),
        'model.hidden_dim': Integer(128, 512)
    }
    
    # Create search
    search = RandomSearch(
        base_config=base_config,
        search_space=search_space,
        metric='val_loss',
        mode='min',
        n_trials=20,
        random_state=42
    )
    
    print(f"\nSearch space:")
    for key, value in search_space.items():
        print(f"  {key}: {value}")
    print(f"Number of trials: 20")
    
    # Run search with early stopping
    best_config, best_score = search.run(
        train_fn=train_vae,
        n_jobs=1,
        experiment_root='experiments/random_search',
        verbose=True,
        early_stopping_patience=5
    )
    
    # Analyze results
    analyzer = SearchAnalyzer(search.results)
    analyzer.print_summary()
    analyzer.plot_convergence(save_path='random_search_convergence.png')
    analyzer.plot_parameter_importance(save_path='random_search_importance.png')
    
    # Save results
    search.save_results('random_search_results.json')
    print(f"\n✓ Results saved to random_search_results.json")
    
    return best_config, best_score


def main():
    print("Hyperparameter Search Example")
    print("=" * 60)
    
    # Run grid search
    grid_best_config, grid_best_score = example_grid_search()
    
    # Run random search
    random_best_config, random_best_score = example_random_search()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\nGrid Search:")
    print(f"  Best val_loss: {grid_best_score:.4f}")
    print(f"  Configuration: {grid_best_config}")
    
    print(f"\nRandom Search:")
    print(f"  Best val_loss: {random_best_score:.4f}")
    print(f"  Configuration: {random_best_config}")
    
    # Determine winner
    if random_best_score < grid_best_score:
        print(f"\n✓ Random search found better configuration!")
        print(f"  Improvement: {(grid_best_score - random_best_score) / grid_best_score * 100:.1f}%")
    else:
        print(f"\n✓ Grid search found better configuration!")
    
    print("\n" + "="*60)
    print("Example complete!")


if __name__ == '__main__':
    main()