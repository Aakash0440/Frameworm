"""
Statistical data drift detection.

Detects when the distribution of incoming data shifts from the
training data distribution — an early warning that model performance
may degrade.

Uses:
- KS test (continuous features)
- Chi-squared test (categorical)
- Maximum Mean Discrepancy (high-dimensional features)

Example:
    >>> detector = DriftDetector()
    >>> detector.fit(train_features)  # Learn reference distribution
    >>> 
    >>> # In production:
    >>> result = detector.check(production_features)
    >>> if result.drift_detected:
    ...     alert("Data drift detected!")
"""

import numpy as np
from typing import Optional, Dict, List
from scipy import stats
from dataclasses import dataclass


@dataclass
class DriftResult:
    """Result of a drift detection check"""
    drift_detected: bool
    p_value: float
    test_statistic: float
    test_name: str
    threshold: float
    
    def __repr__(self):
        status = "⚠️  DRIFT DETECTED" if self.drift_detected else "✓  No drift"
        return (f"DriftResult({status}, p={self.p_value:.4f}, "
                f"test={self.test_name})")


class DriftDetector:
    """
    Statistical drift detector for production monitoring.
    
    Fits on reference data (training set) and checks incoming
    production data against the learned distribution.
    
    Args:
        threshold: p-value threshold for drift detection (default: 0.05)
        test: Statistical test to use:
            'ks'       — Kolmogorov-Smirnov (continuous features, per-dim)
            'mmd'      — Maximum Mean Discrepancy (high-dimensional)
            'chi2'     — Chi-squared (categorical or binned features)
        
    Example:
        >>> detector = DriftDetector(threshold=0.05, test='ks')
        >>> detector.fit(train_features)
        >>> result = detector.check(production_features)
        >>> if result.drift_detected:
        ...     retrain_model()
    """
    
    def __init__(self, threshold: float = 0.05, test: str = 'ks'):
        self.threshold = threshold
        self.test = test
        self.reference_data: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def fit(self, reference: np.ndarray):
        """Learn reference distribution from training data"""
        if not isinstance(reference, np.ndarray):
            reference = np.array(reference)
        self.reference_data = reference.reshape(len(reference), -1)
        self._is_fitted = True
        print(f"✓ DriftDetector fitted on {len(reference)} samples")
        return self
    
    def check(
        self,
        production: np.ndarray,
        return_per_feature: bool = False
    ) -> DriftResult:
        """
        Check production data for drift.
        
        Args:
            production: Production data to check
            return_per_feature: Return per-feature results
            
        Returns:
            DriftResult with overall drift decision
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before check()")
        
        if not isinstance(production, np.ndarray):
            production = np.array(production)
        production = production.reshape(len(production), -1)
        
        if self.test == 'ks':
            return self._ks_test(production)
        elif self.test == 'mmd':
            return self._mmd_test(production)
        elif self.test == 'chi2':
            return self._chi2_test(production)
        else:
            raise ValueError(f"Unknown test: {self.test}")
    
    def _ks_test(self, production: np.ndarray) -> DriftResult:
        """Per-feature Kolmogorov-Smirnov test. Drift if any feature drifts."""
        n_features = min(self.reference_data.shape[1], production.shape[1])
        p_values = []
        
        for i in range(n_features):
            _, p = stats.ks_2samp(
                self.reference_data[:, i],
                production[:, i]
            )
            p_values.append(p)
        
        # Bonferroni correction for multiple comparisons
        min_p = min(p_values) * n_features
        drift = min_p < self.threshold
        
        return DriftResult(
            drift_detected=drift,
            p_value=min_p,
            test_statistic=1.0 - min_p,
            test_name='ks_test',
            threshold=self.threshold
        )
    
    def _mmd_test(self, production: np.ndarray, num_permutations: int = 100) -> DriftResult:
        """Maximum Mean Discrepancy test using RBF kernel."""
        X = self.reference_data
        Y = production
        
        # Subsample for speed
        max_samples = 500
        X = X[:max_samples]
        Y = Y[:max_samples]
        
        # Compute MMD
        def rbf_kernel(A, B, gamma=1.0):
            dists = np.sum((A[:, None] - B[None, :]) ** 2, axis=2)
            return np.exp(-gamma * dists)
        
        XX = rbf_kernel(X, X).mean()
        YY = rbf_kernel(Y, Y).mean()
        XY = rbf_kernel(X, Y).mean()
        mmd = XX + YY - 2 * XY
        
        # Permutation test for p-value
        combined = np.vstack([X, Y])
        n = len(X)
        
        null_mmds = []
        for _ in range(num_permutations):
            idx = np.random.permutation(len(combined))
            X_perm = combined[idx[:n]]
            Y_perm = combined[idx[n:]]
            
            XX_p = rbf_kernel(X_perm, X_perm).mean()
            YY_p = rbf_kernel(Y_perm, Y_perm).mean()
            XY_p = rbf_kernel(X_perm, Y_perm).mean()
            null_mmds.append(XX_p + YY_p - 2 * XY_p)
        
        p_value = float(np.mean(np.array(null_mmds) >= mmd))
        
        return DriftResult(
            drift_detected=p_value < self.threshold,
            p_value=p_value,
            test_statistic=float(mmd),
            test_name='mmd',
            threshold=self.threshold
        )
    
    def _chi2_test(self, production: np.ndarray, n_bins: int = 10) -> DriftResult:
        """Chi-squared test using histogram binning."""
        n_features = min(self.reference_data.shape[1], production.shape[1])
        p_values = []
        
        for i in range(n_features):
            ref_col = self.reference_data[:, i]
            prod_col = production[:, i]
            
            # Create bins from reference
            bins = np.histogram_bin_edges(ref_col, bins=n_bins)
            
            ref_hist, _ = np.histogram(ref_col, bins=bins)
            prod_hist, _ = np.histogram(prod_col, bins=bins)
            
            # Avoid zero counts
            ref_hist = ref_hist + 1e-8
            prod_hist = prod_hist + 1e-8
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            prod_hist = prod_hist / prod_hist.sum()
            
            # Scale to integer counts
            scale = max(len(ref_col), len(prod_col))
            _, p = stats.chisquare(prod_hist * scale, ref_hist * scale)
            p_values.append(p)
        
        min_p = min(p_values) * n_features
        
        return DriftResult(
            drift_detected=min_p < self.threshold,
            p_value=min_p,
            test_statistic=1.0 - min_p,
            test_name='chi2',
            threshold=self.threshold
        )