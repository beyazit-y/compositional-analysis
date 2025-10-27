from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


@dataclass
class ScenarioStats:
    rho: float
    uncertainty: float


class ScenarioBase:
    """
    Handles loading and basic statistics of scenario trace data.
    Computes per-scenario success rates and Hoeffding uncertainty.
    """

    REQUIRED_COLUMNS = {"trace_id", "step", "label"}

    def __init__(self, logbase: Dict[str, str], delta: float = 0.05):
        """
        Args:
            logbase: Dict mapping scenario names to CSV file paths
            delta: Confidence level for Hoeffding bound (default 0.05 → 95% CI)
        """
        self.logbase = logbase
        self.delta = delta
        self.data: Dict[str, pd.DataFrame] = {}

        # Load CSVs
        for name, path in logbase.items():
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"CSV file for scenario '{name}' not found: {path}")
            df = pd.read_csv(path)
            missing = self.REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"CSV for scenario '{name}' missing columns: {missing}")
            df["trace_id"] = df["trace_id"].astype(str)
            self.data[name] = df

        self.success_stats: Dict[str, ScenarioStats] = {}
        self._compute_success_stats()

    def _compute_success_stats(self):
        for name, df in self.data.items():
            last_steps = df.sort_values("step").groupby("trace_id").tail(1)
            labels = last_steps["label"].astype(float).to_numpy()
            rho = labels.mean() if len(labels) > 0 else 0.0
            epsilon = np.sqrt(np.log(2 / self.delta) / (2 * len(labels))) if len(labels) > 0 else 0.0
            self.success_stats[name] = ScenarioStats(rho=rho, uncertainty=epsilon)

    def get_success_rate(self, scenario: str) -> float:
        return self.success_stats[scenario].rho

    def get_success_rate_uncertainty(self, scenario: str) -> float:
        return self.success_stats[scenario].uncertainty


class CompositionalAnalysisEngine:
    """
    Computes importance-sampled success probabilities across sequential scenarios
    using Gaussian KDE and Hoeffding uncertainty propagation.
    """

    def __init__(self, scenario_base: ScenarioBase):
        self.scenario_base = scenario_base

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        """Standardize features along each column (mean=0, std=1)."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return (features - mean) / std

    def analyze(
        self,
        scenario: List[str],
        features: Optional[List[str]] = None,
        norm_feat_idx: Optional[List[int]] = None,
    ) -> Tuple[float, float]:
        """
        Computes importance-sampled success probability and propagated uncertainty.

        Args:
            scenario: Ordered list of scenario names
            features: Optional list of features to include in KDE
            norm_feat_idx: Optional indices of features to normalize

        Returns:
            Tuple of (rho_estimate, uncertainty)
        """
        if len(scenario) == 0:
            raise ValueError("Scenario list must contain at least one scenario.")

        rho = 1.0
        rho_bounds = []

        n = len(scenario)
        delta = self.scenario_base.delta
        per_step_delta = delta / n  # union bound

        for i in range(len(scenario) - 1):
            s_name, t_name = scenario[i], scenario[i+1]
            df_s, df_t = self.scenario_base.data[s_name], self.scenario_base.data[t_name]

            # Select successful endpoints
            s_last = df_s.sort_values("step").groupby("trace_id").tail(1)
            s_last = s_last[s_last["label"] == True]
            t_first = df_t.sort_values("step").groupby("trace_id").head(1)
            t_last = df_t.sort_values("step").groupby("trace_id").tail(1)

            # KDE features (same as your code)
            if features:
                s_features = s_last[features].to_numpy()
                t_features = t_first[features].to_numpy()
                if s_features.shape[0] < 2 or t_features.shape[0] < 2:
                    return 0.0, 0.0
                if norm_feat_idx:
                    for j in norm_feat_idx:
                        s_features[:, j] = self._normalize_features(s_features[:, j].reshape(-1, 1)).flatten()
                        t_features[:, j] = self._normalize_features(t_features[:, j].reshape(-1, 1)).flatten()
            else:
                raise ValueError("Feature list must be provided for KDE.")

            s_features, t_features = s_features.T, t_features.T
            kde_s = gaussian_kde(s_features)
            kde_t = gaussian_kde(t_features)
            p_vals = kde_s(t_features)
            q_vals = kde_t(t_features)
            weights = np.nan_to_num(p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0)

            labels_t_last = t_last["label"].astype(float).to_numpy()
            min_len = min(len(weights), len(labels_t_last))
            weights = weights[:min_len]
            labels_t_last = labels_t_last[:min_len]

            rho_step = np.sum(weights * labels_t_last) / np.sum(weights) if np.sum(weights) > 0 else 0.0
            rho *= rho_step

            # Hoeffding absolute bound with effective samples
            N_eff = np.sum(weights)**2 / np.sum(weights**2) if np.sum(weights**2) > 0 else 1.0
            epsilon_i = np.sqrt(np.log(2 / per_step_delta) / (2 * N_eff))
            rho_bounds.append(epsilon_i)

        # Provable multiplicative error
        prod_factor = np.prod([1 + eps / max(rho_step, 1e-12) for eps in rho_bounds])
        uncertainty = rho * (prod_factor - 1)

        return rho, uncertainty


if __name__ == "__main__":
    logs = {
        "S": "storage/traces/S/traces.csv",
        "X": "storage/traces/X/traces.csv",
        "O": "storage/traces/O/traces.csv",
        "C": "storage/traces/C/traces.csv",
        "SX": "storage/traces/SX/traces.csv",
        "SO": "storage/traces/SO/traces.csv",
        "SC": "storage/traces/SC/traces.csv",
        "SXS": "storage/traces/SXS/traces.csv",
        "SOS": "storage/traces/SOS/traces.csv",
        "SCS": "storage/traces/SCS/traces.csv",
    }
    scenario_base = ScenarioBase(logs)

    print("SMC")
    for s in logs:
        print(f"{s}: rho = {scenario_base.get_success_rate(s):.4f} ± {scenario_base.get_success_rate_uncertainty(s):.4f}")

    engine = CompositionalAnalysisEngine(scenario_base)

    print("Compositional SMC")
    for s in logs:
        rho, uncertainty = engine.analyze(
            s,
            features=["x", "y", "heading", "speed"],
            norm_feat_idx=[0, 1]
        )
        print(f"Estimated {s}: rho = {rho:.4f} ± {uncertainty:.4f}")

