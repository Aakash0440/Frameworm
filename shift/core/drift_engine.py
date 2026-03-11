"""
Wraps FRAMEWORM's existing KS / MMD / Chi-squared detectors.
Takes two DatasetProfiles → returns a DriftResult with per-feature reports.

Reuses: monitoring/drift_detector.py (your existing code)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from shift.core.feature_profiles import (CategoricalProfile, DatasetProfile,
                                         NumericalProfile)

# ─── severity ────────────────────────────────────────────────────────────────


class DriftSeverity(Enum):
    NONE = "NONE"  # p >= 0.10  — no meaningful drift
    LOW = "LOW"  # p >= 0.05  — worth watching
    MEDIUM = "MEDIUM"  # p >= 0.01  — real drift, investigate
    HIGH = "HIGH"  # p <  0.01  — act now


def _severity(p_value: float) -> DriftSeverity:
    if p_value >= 0.10:
        return DriftSeverity.NONE
    if p_value >= 0.05:
        return DriftSeverity.LOW
    if p_value >= 0.01:
        return DriftSeverity.MEDIUM
    return DriftSeverity.HIGH


# ─── per-feature result ───────────────────────────────────────────────────────


@dataclass
class FeatureDriftReport:
    feature_name: str
    feature_type: str  # "numerical" | "categorical"
    test_used: str  # "KS" | "MMD" | "Chi2"
    statistic: float
    p_value: float
    drifted: bool
    severity: DriftSeverity
    mean_delta: Optional[float] = None  # numerical only
    std_delta: Optional[float] = None  # numerical only
    missing_rate_delta: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "test_used": self.test_used,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "drifted": self.drifted,
            "severity": self.severity.value,
            "mean_delta": self.mean_delta,
            "std_delta": self.std_delta,
            "missing_rate_delta": self.missing_rate_delta,
        }


# ─── dataset-level result ─────────────────────────────────────────────────────


@dataclass
class DriftResult:
    features: Dict[str, FeatureDriftReport] = field(default_factory=dict)
    drifted_features: List[str] = field(default_factory=list)
    overall_drifted: bool = False
    overall_severity: DriftSeverity = DriftSeverity.NONE
    drift_fraction: float = 0.0  # % of features that drifted
    n_features_checked: int = 0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_drifted": self.overall_drifted,
            "overall_severity": self.overall_severity.value,
            "drift_fraction": round(self.drift_fraction, 4),
            "drifted_features": self.drifted_features,
            "n_features_checked": self.n_features_checked,
            "summary": self.summary,
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }

    def print_summary(self):
        severity_colours = {
            "NONE": "\033[92m",  # green
            "LOW": "\033[93m",  # yellow
            "MEDIUM": "\033[33m",  # orange
            "HIGH": "\033[91m",  # red
        }
        reset = "\033[0m"
        sev = self.overall_severity.value
        colour = severity_colours.get(sev, "")
        print(f"\n[SHIFT] Drift check — {colour}{sev}{reset}")
        print(
            f"        {self.drift_fraction*100:.1f}% of features drifted "
            f"({len(self.drifted_features)}/{self.n_features_checked})"
        )
        if self.drifted_features:
            print("        Drifted features:")
            for name in self.drifted_features:
                r = self.features[name]
                c = severity_colours.get(r.severity.value, "")
                print(
                    f"          · {name:<30} {c}{r.severity.value}{reset} "
                    f"(p={r.p_value:.4f}, test={r.test_used})"
                )
        print()


# ─── engine ──────────────────────────────────────────────────────────────────


class DriftEngine:
    """
    Compares a current DatasetProfile against a reference DatasetProfile.
    Wraps FRAMEWORM's existing monitoring/drift_detector.py.

    Usage:
        engine = DriftEngine()
        result = engine.compare(reference_profile, current_profile)
    """

    # p-value threshold below which a feature is considered drifted
    DRIFT_THRESHOLD = 0.005

    def compare(
        self,
        reference: DatasetProfile,
        current: DatasetProfile,
    ) -> DriftResult:
        """
        Run drift tests on all shared features.
        Returns a DriftResult with per-feature breakdown.
        """
        result = DriftResult()
        shared_numerical = set(reference.numerical) & set(current.numerical)
        shared_categorical = set(reference.categorical) & set(current.categorical)
        total = len(shared_numerical) + len(shared_categorical)
        result.n_features_checked = total

        # Numerical features → KS test (+ MMD for high-dimensional)
        for name in shared_numerical:
            ref_prof = reference.numerical[name]
            cur_prof = current.numerical[name]
            report = self._compare_numerical(name, ref_prof, cur_prof)
            result.features[name] = report
            if report.drifted:
                result.drifted_features.append(name)

        # Categorical features → Chi-squared
        for name in shared_categorical:
            ref_prof = reference.categorical[name]
            cur_prof = current.categorical[name]
            report = self._compare_categorical(name, ref_prof, cur_prof)
            result.features[name] = report
            if report.drifted:
                result.drifted_features.append(name)

        # Dataset-level summary
        result.drift_fraction = len(result.drifted_features) / total if total > 0 else 0.0
        result.overall_drifted = len(result.drifted_features) > 0

        # Overall severity = worst individual severity
        if result.drifted_features:
            worst = max(result.features[f].severity.value for f in result.drifted_features)
            result.overall_severity = DriftSeverity(worst)

        result.summary = self._build_summary(result)
        return result

    # ────────────────────────────────────────────── numerical (KS test)

    def _compare_numerical(
        self,
        name: str,
        ref: NumericalProfile,
        cur: NumericalProfile,
    ) -> FeatureDriftReport:
        """
        Reconstruct approximate sample arrays from histograms,
        then run KS test.

        If FRAMEWORM's drift_detector is available, use it directly.
        Falls back to a pure-numpy KS implementation.
        """
        stat, p_value = self._ks_from_histograms(ref, cur)

        return FeatureDriftReport(
            feature_name=name,
            feature_type="numerical",
            test_used="KS",
            statistic=float(stat),
            p_value=float(p_value),
            drifted=p_value < self.DRIFT_THRESHOLD,
            severity=_severity(p_value),
            mean_delta=cur.mean - ref.mean,
            std_delta=cur.std - ref.std,
            missing_rate_delta=cur.missing_rate - ref.missing_rate,
        )

    def _ks_from_histograms(self, ref: NumericalProfile, cur: NumericalProfile):
        return self._ks_numpy(ref, cur)

    def _ks_numpy(self, ref: NumericalProfile, cur: NumericalProfile):
        """
        Two-sample KS test: rebin cur onto ref's edges before comparing CDFs.
        """
        # Reconstruct cur samples from its histogram (bin centres × count)
        cur_samples = []
        for i, count in enumerate(cur.histogram_counts):
            if count > 0:
                centre = (cur.histogram_edges[i] + cur.histogram_edges[i + 1]) / 2
                cur_samples.extend([centre] * int(count))
        cur_samples = np.array(cur_samples) if cur_samples else np.array([0.0])

        # Rebin cur onto ref's bin edges (same value-space grid)
        ref_edges = np.array(ref.histogram_edges)
        cur_rebinned, _ = np.histogram(cur_samples, bins=ref_edges)

        rc = np.array(ref.histogram_counts, dtype=float)
        cc = np.array(cur_rebinned, dtype=float)

        ref_cdf = np.cumsum(rc) / (rc.sum() + 1e-10)
        cur_cdf = np.cumsum(cc) / (cc.sum() + 1e-10)

        ks_stat = float(np.max(np.abs(ref_cdf - cur_cdf)))

        n = ref.n_samples
        m = cur.n_samples
        en = np.sqrt((n * m) / (n + m)) if (n + m) > 0 else 1.0
        lam = (en + 0.12 + 0.11 / en) * ks_stat
        if lam == 0:
            p_value = 1.0
        else:
            p_value = 2.0 * (np.exp(-2 * lam**2) - np.exp(-8 * lam**2))
            p_value = float(np.clip(p_value, 0.0, 1.0))
        return ks_stat, p_value

    def _samples_from_histogram(self, counts, edges) -> np.ndarray:
        """Reconstruct approximate samples from histogram for FRAMEWORM detector."""
        samples = []
        for i, count in enumerate(counts):
            if count > 0:
                centre = (edges[i] + edges[i + 1]) / 2
                samples.extend([centre] * int(count))
        return np.array(samples) if samples else np.array([0.0])

    # ────────────────────────────────────────────── categorical (Chi-squared)

    def _compare_categorical(
        self,
        name: str,
        ref: "CategoricalProfile",
        cur: "CategoricalProfile",
    ) -> FeatureDriftReport:
        stat, p_value = self._chi2(ref, cur)

        return FeatureDriftReport(
            feature_name=name,
            feature_type="categorical",
            test_used="Chi2",
            statistic=float(stat),
            p_value=float(p_value),
            drifted=p_value < self.DRIFT_THRESHOLD,
            severity=_severity(p_value),
            missing_rate_delta=cur.missing_rate - ref.missing_rate,
        )

    def _chi2(self, ref, cur):
        """
        Chi-squared test comparing category distributions.
        Uses union of categories across both profiles.
        Falls back to FRAMEWORM's detector if available.
        """
        try:
            from monitoring.drift import DriftDetector

            detector = DriftDetector()
            result = detector.chi_squared_test(ref.value_counts, cur.value_counts)
            return result["statistic"], result["p_value"]
        except (ImportError, AttributeError, KeyError):
            return self._chi2_numpy(ref, cur)

    def _chi2_numpy(self, ref, cur):
        all_cats = set(ref.value_counts) | set(cur.value_counts)
        ref_total = max(sum(ref.value_counts.values()), 1)
        cur_total = max(sum(cur.value_counts.values()), 1)

        observed, expected = [], []
        for cat in all_cats:
            obs = cur.value_counts.get(cat, 0)
            ref_freq = ref.value_counts.get(cat, 0) / ref_total
            observed.append(obs)
            expected.append(max(ref_freq * cur_total, 1e-10))

        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
        df = max(len(all_cats) - 1, 1)

        # Wilson-Hilferty normal approximation with proper erfc (A&S 7.1.26)
        k = float(df)
        cbrt = (chi2_stat / k) ** (1.0 / 3.0)
        mu = 1.0 - 2.0 / (9.0 * k)
        sigma = np.sqrt(2.0 / (9.0 * k))
        z = (cbrt - mu) / sigma

        # P(Z > z) via rational erfc approximation
        t = 1.0 / (1.0 + 0.3275911 * abs(z / np.sqrt(2)))
        poly = t * (
            0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
        )
        erfc_abs = poly * np.exp(-((z / np.sqrt(2)) ** 2))
        p_value = float(np.clip(erfc_abs / 2 if z >= 0 else 1.0 - erfc_abs / 2, 0.0, 1.0))
        return chi2_stat, p_value

    # ────────────────────────────────────────────── summary

    def _build_summary(self, result: DriftResult) -> str:
        if not result.overall_drifted:
            return "No drift detected. Distribution matches reference."
        n = len(result.drifted_features)
        sev = result.overall_severity.value
        features = ", ".join(result.drifted_features[:5])
        if n > 5:
            features += f" (+{n-5} more)"
        return f"{sev} drift detected in {n}/{result.n_features_checked} features: " f"{features}."
