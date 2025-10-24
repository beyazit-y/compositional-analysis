import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpy as np

class Engine:
    def __init__(self, logbase, delta=0.05):
        self.logbase = logbase
        self.data = {name: pd.read_csv(path) for name, path in self.logbase.items()}
        self.delta = delta

        self.success_rates = {}
        self.success_uncertainties = {}
        for name, df in self.data.items():
            last_steps = df.sort_values("step").groupby("trace_id").tail(1)
            labels = last_steps["label"].astype(float).to_numpy()
            rho = labels.mean()
            self.success_rates[name] = rho
            N = len(labels)
            if N > 0:
                epsilon = np.sqrt(np.log(2 / self.delta) / (2 * N))
            else:
                epsilon = 0.0
            self.success_uncertainties[name] = epsilon

    def rho(self, scenario):
        return self.success_rates[scenario]

    def rho_uncertainty(self, scenario):
        return self.success_uncertainties[scenario]

    def analyze(self, scenario, features=None, norm_feat_idx=None):
        """
        Computes importance-sampled success probability and associated uncertainty
        using Hoeffding's inequality. Uncertainty is propagated across sequential scenarios.
        
        Args:
            scenario: list of scenario names
            features: list of features to consider for KDE
            norm_feat_idx: indices of features to normalize
        Returns:
            rho_estimate, uncertainty
        """
        rho = 1.0
        rel_var_squared_sum = 0.0  

        for i in range(len(scenario) - 1):
            s = scenario[i]
            t = scenario[i + 1]

            df_s = self.data[s]
            df_t = self.data[t]

            s_last = df_s.sort_values("step").groupby("trace_id").tail(1)
            s_last = s_last[s_last["label"] == True]

            t_frst = df_t.sort_values("step").groupby("trace_id").head(1)
            t_last = df_t.sort_values("step").groupby("trace_id").tail(1)

            if features is not None:
                s_features = s_last[features].to_numpy()
                t_features = t_frst[features].to_numpy()

            normalize = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)
            if norm_feat_idx is not None:
                for j in norm_feat_idx:
                    s_features[:, j] = normalize(s_features[:, j])
                    t_features[:, j] = normalize(t_features[:, j])

            s_features = s_features.T
            t_features = t_features.T

            kde_s = gaussian_kde(s_features)
            kde_t = gaussian_kde(t_features)

            p_vals = kde_s(t_features)
            q_vals = kde_t(t_features)
            weights = np.nan_to_num(p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0)

            labels_t_last = t_last["label"].astype(float).to_numpy()
            rho_step = np.sum(weights * labels_t_last) / np.sum(weights)
            rho *= rho_step

            N_eff = np.sum(weights)**2 / np.sum(weights**2) # effective sample size
            epsilon_abs = np.sqrt(np.log(2 / self.delta) / (2 * N_eff))
            if rho_step > 0:
                epsilon_rel = epsilon_abs / rho_step
                rel_var_squared_sum += epsilon_rel**2

        uncertainty = rho * np.sqrt(rel_var_squared_sum)
        return rho, uncertainty


if __name__ == "__main__":
    logbase = {
        "S": "storage/traces/S/traces.csv",
        "X": "storage/traces/X/traces.csv",
        "SX": "storage/traces/SX/traces.csv",
        "SXS": "storage/traces/SXS/traces.csv"
    }
    engine = Engine(logbase)
    
    for sc in logbase.keys():
        print(f"{sc}: rho = {engine.rho(sc):.4f} ± {engine.rho_uncertainty(sc):.4f}")

    rho, uncertainty = engine.analyze(
        scenario="SXS",
        features=["x", "y", "heading", "speed"],
        norm_feat_idx=[0, 1]
    )
    print(f"Importance-sampled rho: {rho:.4f} ± {uncertainty:.4f}")

