"""
model/tuner.py
─────────────────────────────────────────────────────────────────────────────
Optuna-based hyperparameter search for XGBoost, optimising PR-AUC.
All trials are logged to MLflow as child runs under the active experiment.
"""

import logging
from typing import Dict, Optional

import mlflow
import numpy as np
import optuna
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    Wraps Optuna study with XGBoost + cross-validated PR-AUC objective.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.tuning_cfg = self.cfg["tuning"]
        self.best_params: Optional[Dict] = None

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scale_pos_weight: float = 1.0,
    ) -> Dict:
        """
        Run Optuna study and return best hyperparameters.

        Parameters
        ----------
        X_train : np.ndarray
            Preprocessed training features
        y_train : np.ndarray
            Binary labels
        scale_pos_weight : float
            Class imbalance weight passed to every trial

        Returns
        -------
        dict of best hyperparameters
        """
        n_pos = int(y_train.sum())
        n_trials = self.tuning_cfg.get("n_trials", 50)

        logger.info(f"\nStarting Optuna search: {n_trials} trials | metric=PR-AUC")

        def objective(trial: optuna.Trial) -> float:
            space = self.tuning_cfg["param_space"]

            params = {
                "n_estimators":      trial.suggest_int("n_estimators", *space["n_estimators"]),
                "max_depth":         trial.suggest_int("max_depth",    *space["max_depth"]),
                "learning_rate":     trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
                "subsample":         trial.suggest_float("subsample",     *space["subsample"]),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
                "min_child_weight":  trial.suggest_int("min_child_weight", *space["min_child_weight"]),
                "gamma":             trial.suggest_float("gamma",     *space["gamma"]),
                "reg_alpha":         trial.suggest_float("reg_alpha", *space["reg_alpha"]),
                "reg_lambda":        trial.suggest_float("reg_lambda",*space["reg_lambda"]),
                "scale_pos_weight":  scale_pos_weight,
                "eval_metric":       "logloss",
                "verbosity":         0,
                "random_state":      42,
                "n_jobs":            -1,
            }

            cv = StratifiedKFold(
                n_splits    = min(self.tuning_cfg.get("cv_folds", 5), n_pos),
                shuffle     = True,
                random_state= 42,
            )

            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                if y_val.sum() == 0:
                    continue

                clf = XGBClassifier(**params)
                clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                y_prob = clf.predict_proba(X_val)[:, 1]
                scores.append(average_precision_score(y_val, y_prob))

            return float(np.mean(scores)) if scores else 0.0

        study = optuna.create_study(
            direction  = "maximize",
            study_name = "iz_defaulter_xgb",
            sampler    = optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        best_val = study.best_value

        logger.info(f"  Best PR-AUC: {best_val:.4f}")
        logger.info(f"  Best params: {self.best_params}")

        # Log to MLflow if active run
        try:
            mlflow.log_params({f"tuned_{k}": v for k, v in self.best_params.items()})
            mlflow.log_metric("tuned_cv_pr_auc", best_val)
        except Exception:
            pass

        return self.best_params
