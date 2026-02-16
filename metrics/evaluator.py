"""
Unified metric evaluation system.
"""

import torch
from typing import Dict, List, Optional, Any, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import FID, InceptionScore, LPIPS


class MetricEvaluator:
    """
    Evaluate generative models with multiple metrics.
    
    Args:
        metrics: List of metric names to compute
        real_data: DataLoader or tensor of real images
        device: Device to use
        batch_size: Batch size for generation
        
    Example:
        >>> evaluator = MetricEvaluator(
        ...     metrics=['fid', 'is'],
        ...     real_data=real_loader,
        ...     device='cuda'
        ... )
        >>> results = evaluator.evaluate(model, num_samples=10000)
    """
    
    def __init__(
        self,
        metrics: List[str],
        real_data: Optional[Any] = None,
        device: str = 'cuda',
        batch_size: int = 100
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.real_data = real_data
        
        # Initialize metrics
        self.metric_objects = {}
        
        for metric_name in metrics:
            metric_name_lower = metric_name.lower()
            
            if metric_name_lower == 'fid':
                self.metric_objects['fid'] = FID(device=device, batch_size=batch_size)
            elif metric_name_lower == 'is':
                self.metric_objects['is'] = InceptionScore(device=device, batch_size=batch_size)
            elif metric_name_lower == 'lpips':
                self.metric_objects['lpips'] = LPIPS(device=device)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
    
    def generate_samples(
        self,
        model: torch.nn.Module,
        num_samples: int,
        generation_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Generate samples from model.
        
        Args:
            model: Generative model
            num_samples: Number of samples to generate
            generation_fn: Custom generation function
                          If None, assumes model has .sample() method
            
        Returns:
            Generated images (N, 3, H, W)
        """
        model.eval()
        
        generated_images = []
        
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating samples"):
                current_batch_size = min(self.batch_size, num_samples - i * self.batch_size)
                
                if generation_fn is not None:
                    batch = generation_fn(model, current_batch_size)
                elif hasattr(model, 'sample'):
                    batch = model.sample(current_batch_size)
                else:
                    raise ValueError(
                        "Model must have .sample() method or provide generation_fn"
                    )
                
                # Ensure in [0, 1] range
                if batch.min() < 0:
                    batch = (batch + 1) / 2  # Convert from [-1, 1] to [0, 1]
                
                generated_images.append(batch.cpu())
        
        return torch.cat(generated_images, dim=0)[:num_samples]
    
    def get_real_images(
        self,
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get real images from dataset.
        
        Args:
            num_samples: Number of samples (if None, use all)
            
        Returns:
            Real images (N, 3, H, W)
        """
        if isinstance(self.real_data, torch.Tensor):
            images = self.real_data
            if num_samples:
                images = images[:num_samples]
            return images
        
        if isinstance(self.real_data, DataLoader):
            images = []
            total = 0
            
            for batch in tqdm(self.real_data, desc="Loading real images"):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                images.append(batch)
                total += len(batch)
                
                if num_samples and total >= num_samples:
                    break
            
            images = torch.cat(images, dim=0)
            
            if num_samples:
                images = images[:num_samples]
            
            return images
        
        raise ValueError("real_data must be Tensor or DataLoader")
    
    def evaluate(
        self,
        model: torch.nn.Module,
        num_samples: int = 10000,
        generation_fn: Optional[Callable] = None,
        save_samples: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model with all metrics.
        
        Args:
            model: Generative model
            num_samples: Number of samples to generate
            generation_fn: Custom generation function
            save_samples: Optional path to save generated samples
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Generate samples
        print(f"Generating {num_samples} samples...")
        generated_images = self.generate_samples(model, num_samples, generation_fn)
        
        # Save if requested
        if save_samples:
            torch.save(generated_images, save_samples)
            print(f"Saved samples to {save_samples}")
        
        # Compute FID
        if 'fid' in self.metric_objects:
            print("\nComputing FID...")
            real_images = self.get_real_images(num_samples)
            fid_score = self.metric_objects['fid'].compute(
                real_images,
                generated_images,
                show_progress=True
            )
            results['fid'] = fid_score
            print(f"FID: {fid_score:.2f}")
        
        # Compute Inception Score
        if 'is' in self.metric_objects:
            print("\nComputing Inception Score...")
            is_score, is_std = self.metric_objects['is'].compute(
                generated_images,
                show_progress=True
            )
            results['is'] = is_score
            results['is_std'] = is_std
            print(f"IS: {is_score:.2f} ± {is_std:.2f}")
        
        # Compute LPIPS (between pairs of generated images)
        if 'lpips' in self.metric_objects:
            print("\nComputing LPIPS...")
            # Take pairs from generated images
            n_pairs = min(1000, len(generated_images) // 2)
            images1 = generated_images[:n_pairs]
            images2 = generated_images[n_pairs:2*n_pairs]
            
            lpips_score = self.metric_objects['lpips'].compute_mean(
                images1,
                images2,
                batch_size=self.batch_size
            )
            results['lpips'] = lpips_score
            print(f"LPIPS: {lpips_score:.4f}")
        
        return results
    
    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        model_class: type,
        config: Any,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model_class: Model class
            config: Model configuration
            **kwargs: Additional evaluation arguments
            
        Returns:
            Metric results
        """
        # Load model
        model = model_class(config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Evaluate
        return self.evaluate(model, **kwargs)


def quick_evaluate(
    model: torch.nn.Module,
    real_data: Any,
    num_samples: int = 5000,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Quick evaluation with default metrics.
    
    Args:
        model: Generative model
        real_data: Real data (Tensor or DataLoader)
        num_samples: Number of samples
        device: Device
        
    Returns:
        Metric results
    """
    evaluator = MetricEvaluator(
        metrics=['fid', 'is'],
        real_data=real_data,
        device=device
    )
    
    return evaluator.evaluate(model, num_samples=num_samples)