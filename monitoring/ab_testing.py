"""
A/B testing for model version comparison.

Routes inference requests between model versions
and collects metrics per version for statistical comparison.

Example:
    >>> ab_test = ABTest(
    ...     model_a=ModelServer('model_v1.pt'),
    ...     model_b=ModelServer('model_v2.pt'),
    ...     split=0.5  # 50/50 traffic split
    ... )
    >>> 
    >>> output = ab_test.predict(input_data)  # Auto-routes
    >>> 
    >>> # After collecting enough data:
    >>> result = ab_test.analyze()
    >>> print(result)  # StatisticalSignificance
"""

import random
import time
from typing import Callable, Optional, Dict, Any
from collections import defaultdict
from scipy import stats as scipy_stats
from dataclasses import dataclass


@dataclass
class ABTestResult:
    """Results of an A/B test analysis"""
    variant_a: str
    variant_b: str
    n_a: int
    n_b: int
    mean_latency_a: float
    mean_latency_b: float
    p_value: float
    significant: bool
    winner: Optional[str]
    
    def __repr__(self):
        winner_str = self.winner or "no winner yet"
        return (f"ABTestResult(winner={winner_str}, "
                f"p={self.p_value:.4f}, "
                f"latency: A={self.mean_latency_a:.1f}ms vs B={self.mean_latency_b:.1f}ms)")


class ABTest:
    """
    A/B test between two model versions.
    
    Routes traffic based on split ratio and collects per-variant metrics.
    Uses two-sample t-test to determine statistical significance.
    
    Args:
        model_a: Callable (inference function) for variant A
        model_b: Callable for variant B
        split: Fraction of traffic to route to B (default: 0.5 = 50/50)
        name_a: Name for variant A
        name_b: Name for variant B
        min_samples: Minimum samples before significance test
    """
    
    def __init__(
        self,
        model_a: Callable,
        model_b: Callable,
        split: float = 0.5,
        name_a: str = 'control',
        name_b: str = 'treatment',
        min_samples: int = 100
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.split = split
        self.name_a = name_a
        self.name_b = name_b
        self.min_samples = min_samples
        
        self._latencies: Dict[str, list] = {name_a: [], name_b: []}
        self._errors: Dict[str, int] = {name_a: 0, name_b: 0}
        self._counts: Dict[str, int] = {name_a: 0, name_b: 0}
    
    def predict(self, *args, **kwargs) -> Any:
        """Route request to A or B based on split ratio"""
        use_b = random.random() < self.split
        variant = self.name_b if use_b else self.name_a
        model = self.model_b if use_b else self.model_a
        
        start = time.perf_counter()
        try:
            result = model(*args, **kwargs)
            self._latencies[variant].append((time.perf_counter() - start) * 1000)
            self._counts[variant] += 1
            return result
        except Exception as e:
            self._errors[variant] += 1
            self._counts[variant] += 1
            raise
    
    def analyze(self) -> ABTestResult:
        """Perform statistical significance test"""
        lats_a = self._latencies[self.name_a]
        lats_b = self._latencies[self.name_b]
        
        if len(lats_a) < self.min_samples or len(lats_b) < self.min_samples:
            print(f"⚠️  Need ≥{self.min_samples} samples per variant "
                  f"(A={len(lats_a)}, B={len(lats_b)})")
        
        import statistics
        mean_a = statistics.mean(lats_a) if lats_a else 0
        mean_b = statistics.mean(lats_b) if lats_b else 0
        
        # Welch's t-test (doesn't assume equal variance)
        if len(lats_a) > 1 and len(lats_b) > 1:
            _, p_value = scipy_stats.ttest_ind(lats_a, lats_b, equal_var=False)
        else:
            p_value = 1.0
        
        significant = p_value < 0.05
        winner = None
        if significant:
            winner = self.name_a if mean_a <= mean_b else self.name_b
        
        return ABTestResult(
            variant_a=self.name_a,
            variant_b=self.name_b,
            n_a=self._counts[self.name_a],
            n_b=self._counts[self.name_b],
            mean_latency_a=mean_a,
            mean_latency_b=mean_b,
            p_value=float(p_value),
            significant=significant,
            winner=winner
        )
    
    def print_summary(self):
        """Print current A/B test status"""
        result = self.analyze()
        print(f"\n🔬 A/B Test: {self.name_a} vs {self.name_b}")
        print(f"  {self.name_a}: {result.n_a} requests, {result.mean_latency_a:.1f}ms avg")
        print(f"  {self.name_b}: {result.n_b} requests, {result.mean_latency_b:.1f}ms avg")
        print(f"  p-value: {result.p_value:.4f} {'(significant!)' if result.significant else ''}")
        if result.winner:
            print(f"  Winner: {result.winner} ✓")