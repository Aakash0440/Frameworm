"""
Experiment visualization tools.
"""

import matplotlib.pyplot as plt
from typing import List, Optional
import pandas as pd
from pathlib import Path


def plot_metric_comparison(
    manager,
    experiment_ids: List[str],
    metric_name: str,
    save_path: Optional[str] = None
):
    """
    Plot metric comparison across experiments.
    
    Args:
        manager: ExperimentManager instance
        experiment_ids: List of experiment IDs
        metric_name: Metric to plot
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    for exp_id in experiment_ids:
        # Get metric history
        df = manager.get_metric_history(exp_id, metric_name)
        
        if len(df) == 0:
            continue
        
        # Get experiment name
        exp = manager.get_experiment(exp_id)
        
        # Plot
        if 'epoch' in df.columns and df['epoch'].notna().any():
            x = df['epoch']
            xlabel = 'Epoch'
        else:
            x = df['step']
            xlabel = 'Step'
        
        plt.plot(x, df['metric_value'], label=exp['name'], marker='o', markersize=3)
    
    plt.xlabel(xlabel)
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def plot_multiple_metrics(
    manager,
    experiment_id: str,
    metric_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot multiple metrics for one experiment.
    
    Args:
        manager: ExperimentManager instance
        experiment_id: Experiment ID
        metric_names: List of metrics to plot
        save_path: Optional path to save figure
    """
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    exp = manager.get_experiment(experiment_id)
    
    for ax, metric_name in zip(axes, metric_names):
        df = manager.get_metric_history(experiment_id, metric_name)
        
        if len(df) == 0:
            continue
        
        # Determine x-axis
        if 'epoch' in df.columns and df['epoch'].notna().any():
            x = df['epoch']
            xlabel = 'Epoch'
        else:
            x = df['step']
            xlabel = 'Step'
        
        # Plot
        ax.plot(x, df['metric_value'], marker='o', markersize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{exp['name']}: {metric_name}")
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def plot_experiment_comparison_grid(
    manager,
    experiment_ids: List[str],
    save_path: Optional[str] = None
):
    """
    Create comparison grid with final metrics.
    
    Args:
        manager: ExperimentManager instance
        experiment_ids: List of experiment IDs
        save_path: Optional path to save figure
    """
    # Get comparison data
    df = manager.compare_experiments(experiment_ids)
    
    # Get numeric columns (metrics)
    metric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    n_metrics = len(metric_cols)
    if n_metrics == 0:
        print("No numeric metrics to plot")
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_cols):
        # Bar plot
        ax.bar(range(len(df)), df[metric])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['name'], rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}')
        ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()