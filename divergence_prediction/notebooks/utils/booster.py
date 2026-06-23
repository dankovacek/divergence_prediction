import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost.callback import LearningRateScheduler

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import json


@dataclass
class Booster:

    """
    Wrapper for quantile XGBoost with:
      - custom CV (predefined folds),
      - tidy logging of trials, learning curves and predictions.

    This class does *not* decide anything about feature groupings, etc.
    You pass in the data, folds, alpha, loss, etc., and then call:
      - run_cv(params)           for a single CV run
    """
    features: Sequence[str]
    target: str
    input_data: pd.DataFrame
    attr_gdf: pd.DataFrame 
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.5, 0.95]))

    loss: str = "reg:quantileerror"
    num_boost_rounds: int = 1000
    device: str = "cuda"
    log_transform_target: bool = True
    trial_seeds: Optional[List[int]] = None
    results_folder: str = "data/results/cv_results"

    # Optional configuration
    id_cols: Optional[Sequence[str]] = None      # e.g. ("donor", "target")

    # Internal state (filled when running CV)
    trial_log: List[Dict[str, Any]] = field(default_factory=list, init=False)


    def _create_fold_dict(self):
        cluster_ids = [int(e) for e in sorted(list(set(self.attr_gdf['5_spatial'].values)))]
        fold_dict = {}
        for c in cluster_ids:
            cluster_stns = self.attr_gdf.loc[self.attr_gdf['5_spatial'] == c, 'official_id'].values
            # in-group edges
            dkl_sample_AND = self.input_data[(self.input_data['donor'].isin(cluster_stns)) & (self.input_data['target'].isin(cluster_stns))].copy()
            # out-of-group edges
            dkl_sample_NOR = self.input_data[(~self.input_data['donor'].isin(cluster_stns)) & (~self.input_data['target'].isin(cluster_stns))].copy()

            # assert that these are mutually exclusive groups
            and_official_ids = set(dkl_sample_AND['donor'].values + dkl_sample_AND['target'].values)
            nor_official_ids = set(dkl_sample_NOR['donor'].values + dkl_sample_NOR['target'].values)
            assert len(list(set(np.intersect1d(and_official_ids, nor_official_ids)))) == 0, 'stations in list are not unique'
            fold_dict[c] = {
                'test': dkl_sample_AND.index.values,
                'train': dkl_sample_NOR.index.values,
            }
        self.fold_dict = fold_dict

    def _build_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Translate Hyperopt search variables into XGBoost params."""
        obj = {
            "objective": self.loss,
            "eta": 10 ** params["log10_eta"],
            # "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            # "gamma": 10 ** params["log10_gamma"],
            "device": self.device,
            "sampling_method": "gradient_based",
            "tree_method": "hist",
            "max_depth": params["max_depth"],
        }
        if self.loss == 'reg:quantileerror':
            obj["quantile_alpha"] = self.alpha
        return obj

    def _prepare_fold_data(
        self,
        cv_indices: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Slice input_data into train/test arrays for a given fold.
        Returns X_train, X_test, y_train, y_test, test_idx.
        """

        test_idx = cv_indices["test"]
        train_idx = cv_indices["train"]

        train_df = self.input_data.iloc[train_idx, :]
        test_df = self.input_data.iloc[test_idx, :]

        X_train = train_df[self.features].to_numpy(dtype="float32", copy=False)
        X_test = test_df[self.features].to_numpy(dtype="float32", copy=False)

        y_train_raw = train_df[self.target].values
        y_test_raw = test_df[self.target].values

        if self.log_transform_target:
            y_train = np.log(y_train_raw)
            y_test = np.log(y_test_raw)
        else:
            y_train = y_train_raw
            y_test = y_test_raw

        return X_train, X_test, y_train, y_test, test_idx

    
    def _lr_decay(self, boosting_round):
        if boosting_round < 1000:
            return 0.01
        elif boosting_round < 1500:
            return 0.005
        else:
            return 0.001
    
    
    def _train_single_fold(
        self,
        fold_no: int,
        params: Dict[str, Any],
        cv_indices: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        X_train, X_test, y_train, y_test, test_idx = self._prepare_fold_data(cv_indices)

        if self.loss == 'reg:quantileerror':
            dtrain = xgb.QuantileDMatrix(X_train, y_train, feature_names=list(self.features))
            dtest = xgb.QuantileDMatrix(X_test, y_test, ref=dtrain, feature_names=list(self.features))
        else:
            dtrain = xgb.DMatrix(X_train, y_train, feature_names=list(self.features))
            dtest = xgb.DMatrix(X_test, y_test, feature_names=list(self.features))

        evals_result: Dict[str, Dict[str, List[float]]] = {}
        eval_list = [(dtrain, "train"), (dtest, "test")]

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_rounds,
            evals=eval_list,
            evals_result=evals_result,
            verbose_eval=0,
            callbacks=[LearningRateScheduler(self._lr_decay)],
        )

        eval_keys = list(evals_result["train"].keys())
        eval_key = eval_keys[0]

        train_perf = np.asarray(evals_result["train"][eval_key], dtype=float)
        test_perf = np.asarray(evals_result["test"][eval_key], dtype=float)

        # learning curve for this fold
        fold_progress = pd.DataFrame(
            {
                "fold": fold_no,
                "round": np.arange(len(train_perf)),
                "train": train_perf,
                "test": test_perf,
            }
        )

        # predictions
        preds = booster.predict(dtest)  # (n_samples, n_quantiles)
        assert preds.shape[0] == X_test.shape[0]
        if self.loss == 'reg:quantileerror':
            assert preds.shape[1] == self.alpha.shape[0]

        fold_pred = pd.DataFrame(
            {
                "row_idx": test_idx,
                "fold": fold_no,
                "actual": y_test,
            }
        )
        if self.loss == 'reg:quantileerror':
            for j, a in enumerate(self.alpha):
                q_name = f"q{int(round(100 * a))}"  # e.g. q5, q50, q95
                fold_pred[q_name] = preds[:, j]
        else:
            fold_pred["prediction"] = preds

        if self.id_cols is not None and all(
            c in self.input_data.columns for c in self.id_cols
        ):
            ids = (
                self.input_data
                .loc[test_idx, list(self.id_cols)]
                .astype(str)
                .agg("_".join, axis=1)
            )
            fold_pred["station_pair"] = ids

        # --- NEW: per-fold feature importance (total_gain; normalized) ---
        raw_imp = booster.get_score(importance_type="total_gain")  # dict: {feature_name: gain}
        feat_names = list(self.features)  # ensure consistent order
        vals = np.array([raw_imp.get(f, 0.0) for f in feat_names], dtype=float)
        denom = vals.sum()
        if denom > 0:
            vals /= denom
        fold_importance = pd.DataFrame(
            {"feature": feat_names, "importance_total_gain": vals, "fold": fold_no}
        )

        return train_perf, test_perf, fold_progress, fold_pred, fold_importance

    def run_cv(
        self,
        params: Dict[str, Any],
    ) -> Tuple[float, float, float, pd.DataFrame, pd.DataFrame]:
        """
        Run custom CV once for a fixed set of params.
        Saves per-fold and aggregated CSVs with trial + fold tags.
        """
        fold_best_test: List[float] = []
        fold_progress_list: List[pd.DataFrame] = []
        fold_pred_list: List[pd.DataFrame] = []
        fold_imp_list: List[pd.DataFrame] = []

        self._create_fold_dict()
        os.makedirs(self.results_folder, exist_ok=True)

        trial_no = len(self.trial_log) + 1  # or pass in explicitly if preferred

        for fold_no, cv_indices in self.fold_dict.items():

            train_perf, test_perf, fold_progress, fold_pred, fold_imp = self._train_single_fold(
                fold_no=fold_no,
                params=params,
                cv_indices=cv_indices,
            )

            # tag with identifiers
            fold_progress = fold_progress.copy()
            fold_progress["trial"] = trial_no
            fold_progress["fold"] = fold_no

            fold_pred = fold_pred.copy()
            fold_pred["trial"] = trial_no
            fold_pred["fold"]  = fold_no
            
            fold_imp['trial'] = trial_no
            fold_imp['fold']  = fold_no

            # (optional) long-form learning curves: split=train/test
            # comment this block out if you prefer wide form
            fold_progress_long = fold_progress.melt(
                id_vars=["trial", "fold", "round"],
                value_vars=["train", "test"],
                var_name="split",
                value_name="loss",
            )

            fold_best_test.append(float(np.min(test_perf)))
            fold_progress_list.append(fold_progress_long)  # store long-form
            fold_pred_list.append(fold_pred)
            fold_imp_list.append(fold_imp)

        # Summary statistics
        best_test_arr = np.asarray(fold_best_test, dtype=float)
        mean_test = float(best_test_arr.mean())
        median_test = float(np.median(best_test_arr))
        stdev_test = float(best_test_arr.std(ddof=0))

        # Aggregated (already trial/fold-tagged)
        learning_curves_df = pd.concat(fold_progress_list, ignore_index=True)
        predictions_df     = pd.concat(fold_pred_list,     ignore_index=True)
        feature_importance_df = pd.concat(fold_imp_list,    ignore_index=True)

        # Save aggregated CSVs
        learning_curves_df.to_csv(
            os.path.join(self.results_folder, f"trial_{trial_no:03d}_all_folds_progress.csv"),
            index=False,
        )
        predictions_df.to_csv(
            os.path.join(self.results_folder, f"trial_{trial_no:03d}_all_folds_predictions.csv"),
            index=False,
        )

        feature_importance_df.to_csv(
            os.path.join(self.results_folder, f"trial_{trial_no:03d}_all_folds_feature_importance.csv"),
            index=False,
        )

        return mean_test, median_test, stdev_test, learning_curves_df, predictions_df, feature_importance_df

