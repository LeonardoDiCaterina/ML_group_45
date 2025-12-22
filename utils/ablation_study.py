import os
import numpy as np
import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score
)

warnings.filterwarnings("ignore")



# ---------------------------------------------------------------------
# Regression metrics helper
# ---------------------------------------------------------------------
def compute_regression_metrics(y_true, y_pred, n_features: int):
    """Compute the metrics used in the project ablation pipeline.

    Returns a dict with keys:
    r2, adjusted_r2, rmse, mae, mse, mape
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)

    # mape: sklearn handles zeros by returning inf; we guard manually.
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

    r2 = r2_score(y_true, y_pred)

    # adjusted R2 uses n_features from the *raw* feature set used in this run
    n = len(y_true)
    p = int(n_features)
    if n - p - 1 > 0:
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    else:
        adjusted_r2 = np.nan

    return {
        "r2": float(r2),
        "adjusted_r2": float(adjusted_r2) if adjusted_r2 == adjusted_r2 else np.nan,
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "mape": float(mape) if mape == mape else np.nan,
    }


# ---------------------------------------------------------------------
# Cross-validated ablation runner
# ---------------------------------------------------------------------
def run_ablation_cv(
    X: pd.DataFrame,
    y,
    model=None,
    builder=None,
    cv_folds: int = 5,
    random_state: int = 42,
    features_to_test=None,
    baseline_name: str = "Baseline",
    verbose: bool = True,
):
    """Run k-fold CV ablation (remove 1 feature at a time) and return a results DataFrame.

    This produces the columns: model_variant, features_removed, fold, r2, adjusted_r2, rmse, mae, mse, mape

    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame so we can drop features by name.")

    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)

    all_cols = list(X.columns)
    if features_to_test is None:
        features_to_test = all_cols
    else:
        features_to_test = [f for f in features_to_test if f in all_cols]

    if model is None and builder is None:
        raise ValueError("Provide either 'model' or 'builder'.")
    if model is not None and builder is not None:
        raise ValueError("Provide only one of 'model' or 'builder', not both.")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows = []

    def _make_estimator(use_cols):
        if builder is not None:
            return builder(use_cols)
        return clone(model)

    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        # Baseline
        base_cols = all_cols
        est = _make_estimator(base_cols)
        est.fit(X_tr[base_cols], y_tr)
        pred = est.predict(X_va[base_cols])
        metrics = compute_regression_metrics(y_va, pred, n_features=len(base_cols))
        rows.append({
            "model_variant": baseline_name,
            "features_removed": "none",
            "fold": fold,
            **metrics
        })

        # Ablations
        for feat in features_to_test:
            use_cols = [c for c in all_cols if c != feat]
            est = _make_estimator(use_cols)
            est.fit(X_tr[use_cols], y_tr)
            pred = est.predict(X_va[use_cols])
            metrics = compute_regression_metrics(y_va, pred, n_features=len(use_cols))
            rows.append({
                "model_variant": f"drop_{feat}",
                "features_removed": feat,
                "fold": fold,
                **metrics
            })

        if verbose:
            elapsed = time.time() - start
            print(f"[ablation_cv] fold {fold}/{cv_folds} done ({elapsed:.1f}s elapsed)")

    return pd.DataFrame(rows)


class ProgressiveAblationSelector:
    """
    Progressive Ablation Study for feature selection and model interpretability.

    This class implements a two-stage ablation framework:

    1) Initial Ablation (One-Feature-at-a-Time)
       - Each feature is removed individually
       - The performance degradation is measured using cross-validation
       - Features are ranked by their average performance drop

    2) Progressive Ablation on Top-Ranked Features
       - Feature subsets are evaluated following a decreasing strategy:
         All → Top 6 → Top 4 → Top 3 → Top 2 → Top 1
       - For each subset size, all possible combinations are tested
       - The best-performing subset is retained

    The framework is model-agnostic and supports both regression and
    classification tasks via metric selection.
    """

    def __init__(
        self,
        model,
        metric: str = "r2",
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initializes the ProgressiveAblationSelector.

        Parameters
        ----------
        model : sklearn estimator
            Any scikit-learn compatible estimator implementing
            `fit()` and `predict()`.

        metric : str, default="r2"
            Evaluation metric used during ablation.
            Supported options:
            - "r2"       : Coefficient of determination (regression)
            - "mse"      : Mean Squared Error (regression, lower is better)
            - "accuracy" : Accuracy score (classification)

        cv_folds : int, default=5
            Number of folds for K-Fold cross-validation.

        random_state : int, default=42
            Random seed to ensure reproducibility of cross-validation splits.

        verbose : bool, default=True
            If True, prints progress and intermediate information during execution.
        """
        self.model = model
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

        # Stores ranked feature importance after initial ablation
        self.feature_importance_df = None

        # Stores all progressive ablation results
        self.progressive_results = []

    # ------------------------------------------------------------------
    # Metric handling
    # ------------------------------------------------------------------
    def _calculate_metric(self, y_true, y_pred):
        """
        Computes the selected evaluation metric.

        Parameters
        ----------
        y_true : array-like
            Ground truth target values.

        y_pred : array-like
            Predicted target values.

        Returns
        -------
        float
            Metric value according to the selected metric.
        """
        if self.metric == "r2":
            return r2_score(y_true, y_pred)
        elif self.metric == "mse":
            return mean_squared_error(y_true, y_pred)
        elif self.metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    # ------------------------------------------------------------------
    # Step 1: Initial Ablation
    # ------------------------------------------------------------------
    def perform_initial_ablation(self, X, y, feature_names=None):
        """
        Performs an initial ablation study by removing one feature at a time.

        For each fold:
        - Trains a baseline model using all features
        - Trains ablated models removing one feature at a time
        - Computes performance degradation caused by each removal

        Feature importance is defined as the average performance drop
        across all folds.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        y : np.ndarray
            Target vector.

        feature_names : list of str, optional
            Names of the features. If None, generic names are generated.

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature importance ranking with columns:
            - feature
            - importance
            - abs_importance
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # Store performance drops per feature
        fold_results = {f: [] for f in feature_names}
        baseline_scores = []

        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 1: INITIAL ABLATION STUDY")
            print("=" * 70)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Baseline model with all features
            base_model = clone(self.model)
            base_model.fit(X_train, y_train)
            y_pred_base = base_model.predict(X_val)
            baseline_score = self._calculate_metric(y_val, y_pred_base)
            baseline_scores.append(baseline_score)

            # Remove each feature individually
            for i, feature in enumerate(feature_names):
                X_train_drop = np.delete(X_train, i, axis=1)
                X_val_drop = np.delete(X_val, i, axis=1)

                model = clone(self.model)
                model.fit(X_train_drop, y_train)
                y_pred = model.predict(X_val_drop)

                score = self._calculate_metric(y_val, y_pred)
                performance_drop = baseline_score - score
                fold_results[feature].append(performance_drop)

        # Aggregate importance across folds
        feature_importance = {
            f: np.mean(v) for f, v in fold_results.items()
        }

        self.feature_importance_df = (
            pd.DataFrame({
                "feature": feature_importance.keys(),
                "importance": feature_importance.values(),
                "abs_importance": [abs(v) for v in feature_importance.values()]
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Store baseline result
        self.progressive_results.append({
            "strategy": "all_features",
            "n_features": len(feature_names),
            "features": feature_names,
            "mean_score": np.mean(baseline_scores),
            "std_score": np.std(baseline_scores),
            "scores": baseline_scores
        })

        return self.feature_importance_df

    # ------------------------------------------------------------------
    # Step 2: Progressive Ablation
    # ------------------------------------------------------------------
    def progressive_ablation_analysis(self, X, y, feature_names=None):
        """
        Performs progressive ablation using subsets of top-ranked features.

        The method evaluates decreasing feature sets:
        All → Top 6 → Top 4 → Top 3 → Top 2 → Top 1

        For each subset size:
        - All possible feature combinations are evaluated
        - The best-performing subset is retained

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        y : np.ndarray
            Target vector.

        feature_names : list of str, optional
            Feature names corresponding to X columns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all progressive ablation results,
            sorted by performance.
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        if self.feature_importance_df is None:
            self.perform_initial_ablation(X, y, feature_names)

        n_features = len(feature_names)

        if n_features >= 6:
            top_n_list = [6, 4, 3, 2, 1]
        elif n_features >= 4:
            top_n_list = [4, 3, 2, 1]
        elif n_features >= 3:
            top_n_list = [3, 2, 1]
        else:
            top_n_list = [1]

        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        for top_n in top_n_list:
            top_features = (
                self.feature_importance_df
                .head(top_n)["feature"]
                .tolist()
            )
            indices = [feature_names.index(f) for f in top_features]

            best_score = -np.inf if self.metric != "mse" else np.inf
            best_combo = None

            for r in range(1, top_n + 1):
                for combo in combinations(indices, r):
                    X_subset = X[:, combo]
                    scores = []
                    start = time.time()

                    for tr, va in kf.split(X_subset):
                        model = clone(self.model)
                        model.fit(X_subset[tr], y[tr])
                        y_pred = model.predict(X_subset[va])
                        scores.append(
                            self._calculate_metric(y[va], y_pred)
                        )

                    mean_score = np.mean(scores)
                    std_score = np.std(scores)

                    better = (
                        mean_score < best_score
                        if self.metric == "mse"
                        else mean_score > best_score
                    )

                    if better:
                        best_score = mean_score
                        best_combo = {
                            "features": [feature_names[i] for i in combo],
                            "scores": scores,
                            "mean_score": mean_score,
                            "std_score": std_score,
                            "elapsed_time": time.time() - start
                        }

            if best_combo:
                self.progressive_results.append({
                    "strategy": f"top_{top_n}_best_combo",
                    "n_features": len(best_combo["features"]),
                    "features": best_combo["features"],
                    "mean_score": best_combo["mean_score"],
                    "std_score": best_combo["std_score"],
                    "scores": best_combo["scores"],
                    "elapsed_time": best_combo["elapsed_time"]
                })

        df = pd.DataFrame(self.progressive_results)
        return (
            df.sort_values("mean_score")
            if self.metric == "mse"
            else df.sort_values("mean_score", ascending=False)
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_optimal_features(self, top_k=1):
        """
        Returns the top-performing feature subsets.

        Parameters
        ----------
        top_k : int, default=1
            Number of best-performing subsets to return.

        Returns
        -------
        list of dict
            Each dictionary contains:
            - strategy
            - n_features
            - features
            - mean_score
            - std_score
        """
        df = pd.DataFrame(self.progressive_results)
        df = (
            df.sort_values("mean_score")
            if self.metric == "mse"
            else df.sort_values("mean_score", ascending=False)
        )

        return df.head(top_k).to_dict(orient="records")
