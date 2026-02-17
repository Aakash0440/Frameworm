"""
Analysis tools for hyperparameter search results.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SearchAnalyzer:
    """
    Analyze hyperparameter search results.

    Args:
        results: List of result dictionaries from search

    Example:
        >>> analyzer = SearchAnalyzer(search.results)
        >>> analyzer.plot_convergence()
        >>> analyzer.print_summary()
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.df = self._results_to_dataframe()

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []

        for result in self.results:
            row = {"trial_idx": result["trial_idx"], "score": result["score"]}

            # Add config parameters
            for key, value in result["config"].items():
                row[key] = value

            # Add all metrics
            for key, value in result["metrics"].items():
                row[f"metric_{key}"] = value

            data.append(row)

        return pd.DataFrame(data)

    def get_best_n(self, n: int = 5) -> pd.DataFrame:
        """Get top N configurations"""
        return self.df.nsmallest(n, "score")

    def plot_convergence(self, save_path: Optional[str] = None, show_best: bool = True):
        """
        Plot convergence over trials.

        Args:
            save_path: Path to save figure
            show_best: Show running best score
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all scores
        ax.plot(self.df["trial_idx"], self.df["score"], "o-", alpha=0.6, label="Trial score")

        # Plot running best
        if show_best:
            running_best = self.df["score"].cummin()
            ax.plot(self.df["trial_idx"], running_best, "r-", linewidth=2, label="Best so far")

        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.set_title("Search Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()

    def plot_parameter_importance(self, save_path: Optional[str] = None, top_n: int = 10):
        """
        Plot parameter importance based on correlation with score.

        Args:
            save_path: Path to save figure
            top_n: Number of top parameters to show
        """
        # Get config columns
        config_cols = [
            col
            for col in self.df.columns
            if not col.startswith("metric_") and col not in ["trial_idx", "score"]
        ]

        # Compute correlations
        correlations = {}
        for col in config_cols:
            # Convert to numeric if possible
            try:
                values = pd.to_numeric(self.df[col])
                corr = abs(values.corr(self.df["score"]))
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                pass

        if not correlations:
            print("No numeric parameters to analyze")
            return

        # Sort by importance
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        params, importances = zip(*sorted_params)
        ax.barh(range(len(params)), importances)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_xlabel("Absolute Correlation with Score")
        ax.set_title("Parameter Importance")
        ax.grid(True, alpha=0.3, axis="x")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()

    def plot_parameter_vs_score(self, param_name: str, save_path: Optional[str] = None):
        """
        Plot parameter value vs score.

        Args:
            param_name: Parameter name
            save_path: Path to save figure
        """
        if param_name not in self.df.columns:
            raise ValueError(f"Parameter '{param_name}' not found")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(self.df[param_name], self.df["score"], alpha=0.6)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Score")
        ax.set_title(f"{param_name} vs Score")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("SEARCH SUMMARY")
        print("=" * 60)

        print(f"\nTotal trials: {len(self.results)}")
        print(f"Best score: {self.df['score'].min():.4f}")
        print(f"Worst score: {self.df['score'].max():.4f}")
        print(f"Mean score: {self.df['score'].mean():.4f}")
        print(f"Std score: {self.df['score'].std():.4f}")

        print(f"\nTop 3 configurations:")
        top3 = self.get_best_n(3)
        for i, row in top3.iterrows():
            print(f"\n  Rank {i+1} (score={row['score']:.4f}):")
            config_cols = [
                col
                for col in self.df.columns
                if not col.startswith("metric_") and col not in ["trial_idx", "score"]
            ]
            for col in config_cols:
                print(f"    {col}: {row[col]}")

        print("\n" + "=" * 60)
