# import pandas as pd

# class Engine:
#     def __init__(self):
#         self.scenario = "SX"
#         self.logs = {
#             "S": "storage/traces/S/traces.csv",
#             "X": "storage/traces/X/traces.csv"
#         }

#         self.data = {name: pd.read_csv(path) for name, path in self.logs.items()}

#         self.rho = {}
#         for name, df in self.data.items():
#             last_steps = df.sort_values("step").groupby("trace_id").tail(1)
#             success_rate = 100 * last_steps["label"].mean()
#             self.rates[name] = success_rate

#     def analyze(self):
#         # TODO: Do compositional analysis, i.e., compute rho for the composite scenario "SX" using the resutls for "S" and "X"
#         # To do so, compute the empirical distribution over x,y,heading,speed,action,reward fields for scenario "S" and then do importance sampling (using this) for the results of "X" and combine rhos based on this

# import pandas as pd
# from scipy.stats import gaussian_kde
# import numpy as np

# class Engine:
#     def __init__(self):
#         self.scenario = "SX"
#         self.logs = {
#             # "S": "storage/traces/S/traces.csv",
#             # "X": "storage/traces/X/traces.csv",
#             "S": "temp/traces_half0.csv",
#             "X": "temp/traces_half1.csv",
#             "SX": "storage/traces/SX/traces.csv"
#         }

#         self.data = {name: pd.read_csv(path) for name, path in self.logs.items()}

#         # Store success rates per scenario
#         self.rates = {}
#         for name, df in self.data.items():
#             last_steps = df.sort_values("step").groupby("trace_id").tail(1)
#             success_rate = 100 * last_steps["label"].mean()
#             self.rates[name] = success_rate

#         self.rho = {}

#     def analyze(self):
#         features = ["x", "y", "heading", "speed", "reward"]

#         df_S = self.data["S"]
#         df_X = self.data["X"]

#         S_last = df_S.sort_values("step").groupby("trace_id").tail(1)
#         S_last = S_last[S_last["label"] == True]

#         X_frst = df_X.sort_values("step").groupby("trace_id").head(1)
#         X_last = df_X.sort_values("step").groupby("trace_id").tail(1)

#         S_last_features = S_last[features].to_numpy().T
#         X_frst_features = X_frst[features].to_numpy().T

#         p_S = gaussian_kde(S_last_features)
#         q_X = gaussian_kde(X_frst_features)

#         p_vals = p_S(X_frst_features)
#         q_vals = q_X(X_frst_features)
#         weights = np.nan_to_num(p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0)

#         labels_X_last = X_last["label"].astype(float).to_numpy()
#         rho_SX = np.sum(weights * labels_X_last) / np.sum(weights)

#         return rho_SX

import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpy as np

class Engine:
    def __init__(self):
        self.scenario = "SX"
        self.logs = {
            "S": "storage/traces/S/traces.csv",
            "X": "storage/traces/X/traces.csv",
            # "S": "temp/traces_half0.csv",
            # "X": "temp/traces_half1.csv",
            "SX": "storage/traces/SX/traces.csv"
        }

        self.data = {name: pd.read_csv(path) for name, path in self.logs.items()}

        # Store success rates per scenario
        self.rates = {}
        for name, df in self.data.items():
            last_steps = df.sort_values("step").groupby("trace_id").tail(1)
            success_rate = 100 * last_steps["label"].mean()
            self.rates[name] = success_rate

        print(self.rates)

        self.rho = {}

    def analyze(self):
        features = ["x", "y", "heading", "speed"]

        df_S = self.data["S"]
        df_X = self.data["X"]

        # Get last step of S traces that succeeded
        S_last = df_S.sort_values("step").groupby("trace_id").tail(1)
        S_last = S_last[S_last["label"] == True]

        # Get first and last steps of X traces
        X_frst = df_X.sort_values("step").groupby("trace_id").head(1)
        X_last = df_X.sort_values("step").groupby("trace_id").tail(1)

        # Extract features
        S_features = S_last[features].to_numpy()
        X_features = X_frst[features].to_numpy()

        normalize = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)

        S_features[:, 0] = normalize(S_features[:, 0])
        S_features[:, 1] = normalize(S_features[:, 1])

        X_features[:, 0] = normalize(X_features[:, 0])
        X_features[:, 1] = normalize(X_features[:, 1])

        S_features = S_features.T
        X_features = X_features.T

        # # Apply PCA to reduce dimensionality
        # pca = PCA(n_components=2)
        # S_pca = pca.fit_transform(S_features)
        # X_pca = pca.transform(X_features)

        # # print(S_pca)
        # # print(X_pca)
        # # input()

        # # Keep only nonzero-variance components
        # nonzero_var = np.where(pca.explained_variance_ > 1e-10)[0]
        # S_pca = S_pca[:, nonzero_var].T  # KDE expects shape (dim, n_samples)
        # X_pca = X_pca[:, nonzero_var].T

        # # Fit KDEs
        # kde_S = gaussian_kde(S_pca)
        # kde_X = gaussian_kde(X_pca)

        # Fit KDEs
        kde_S = gaussian_kde(S_features)
        kde_X = gaussian_kde(X_features)

        # Importance sampling weights
        p_vals = kde_S(X_features)
        q_vals = kde_X(X_features)
        weights = np.nan_to_num(p_vals / q_vals, nan=0.0, posinf=0.0, neginf=0.0)

        # Weighted success rate
        labels_X_last = X_last["label"].astype(float).to_numpy()
        rho_SX = np.sum(weights * labels_X_last) / np.sum(weights)

        return rho_SX


if __name__ == "__main__":
    engine = Engine()
    rho = engine.analyze()
    print(rho)
