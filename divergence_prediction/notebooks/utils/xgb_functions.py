import numpy as np
import pandas as pd
import xgboost as xgb

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def compute_target_bins(Y, q=20):
    return pd.qcut(Y, q=q, labels=False, duplicates='drop')


def generate_random_params(n_trials, seed=42):
    rng = np.random.default_rng(seed)
    # core rates
    learning_rate = 10 ** rng.uniform(-2.3, -1.0, n_trials)    # ~0.005 .. 0.1
    subsample     = rng.uniform(0.7, 1.0, n_trials)
    colsample     = rng.uniform(0.7, 1.0, n_trials)

    # capacity & regularization
    max_depth         = rng.integers(2, 7, n_trials)         # 2..6 inclusive
    min_child_weight  = 10 ** rng.uniform(0.0, 1.5, n_trials)  # ~1 .. 32
    reg_lambda        = 10 ** rng.uniform(-0.5, 1.5, n_trials) # ~0.3 .. 32
    reg_alpha         = 10 ** rng.uniform(-2.0, 1.0, n_trials) # ~0.01 .. 10
    gamma             = 10 ** rng.uniform(-2.0, 0.7, n_trials) # ~0.01 .. 5

    params = []
    for i in range(n_trials):
        params.append(dict(
            # training / tree builder
            # rates / sampling
            learning_rate=float(learning_rate[i]),
            subsample=float(subsample[i]),
            colsample_bytree=float(colsample[i]),

            # capacity & regs
            max_depth=int(max_depth[i]),
            min_child_weight=float(min_child_weight[i]),
            reg_lambda=float(reg_lambda[i]),
            reg_alpha=float(reg_alpha[i]),
            gamma=float(gamma[i]),
        ))
    return params

def create_feature_map(features, fmap_path):
    with open(fmap_path, 'w') as f:
        for i, feat in enumerate(features):
            f.write(f"{i}\t{feat}\tq\n")


def save_feature_importance(model, features, output_path):
    importance_df = pd.DataFrame(
        {
            "feature": features,
            "gain": [model.get_score(importance_type="gain").get(feature, 0.0) for feature in features],
            "weight": [model.get_score(importance_type="weight").get(feature, 0.0) for feature in features],
            "cover": [model.get_score(importance_type="cover").get(feature, 0.0) for feature in features],
        }
    )
    importance_df.sort_values("gain", ascending=False).to_csv(output_path, index=False)


def train_single_fold(X_train, X_test, Y_train, Y_test, params, loss, num_boost_rounds, importance_output_path, features):
    if loss == 'reg:quantileerror':
        params['quantile_alpha'] = np.array([0.05, 0.5, 0.95])
        dtrain = xgb.QuantileDMatrix(X_train, label=Y_train)
        dtest = xgb.QuantileDMatrix(X_test, label=Y_test)
    else:
        dtrain = xgb.DMatrix(X_train, label=Y_train)
        dtest = xgb.DMatrix(X_test, label=Y_test)

    evals_result = {}
    eval_list = [(dtrain, "train"), (dtest, "eval")]
    
    model = xgb.train(
        params, dtrain, num_boost_round=num_boost_rounds, 
        evals=eval_list, evals_result=evals_result, verbose_eval=0
    )
    eval_key = list(evals_result['train'].keys())[0]
    train_perf = np.array(evals_result['train'][eval_key])
    test_perf = np.array(evals_result['eval'][eval_key])
    predictions = model.predict(dtest)

    save_feature_importance(model, features, importance_output_path)

    return predictions, train_perf, test_perf, eval_key


def run_xgb_CV_trials(
    set_name, features, target, input_data,
    n_optimization_rounds, nfolds, num_boost_rounds,
    results_folder, loss='reg:squarederror', random_seed=42
):
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    
    X_input = input_data[features].values
    Y_input = input_data[target].values
    y_binned = compute_target_bins(Y_input)
    # stratified k-fold cross-validation ensures that each fold has a similar distribution of target values
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_seed)

    # generate xgboost hyperparameters for learning rate, subsample, and colsample_bytree
    hyperparameter_sets = generate_random_params(n_optimization_rounds, random_seed)

    all_trial_results = []
    all_predictions = []
    all_convergence_data = []

    for trial in range(n_optimization_rounds):
        hyperparameters = hyperparameter_sets[trial]
        params = {
            "objective": loss,
            "seed": random_seed,
            "device": "cuda",
            "sampling_method": "gradient_based",
            "validate_parameters": True,
            **hyperparameters 
        }

        fold_results = []
        fold_perfs = []
        convergence_data = []
        for fold_no, (train_idx, test_idx) in enumerate(skf.split(X_input, y_binned)):
            X_train, X_test = X_input[train_idx], X_input[test_idx]
            Y_train, Y_test = Y_input[train_idx], Y_input[test_idx]
            test_ids = input_data.iloc[test_idx]['official_id'].values

            feature_importance_path = Path(results_folder) / 'feature_importance' / f"{set_name}_trial{trial}_fold{fold_no}_feature_importance.csv"
            if not feature_importance_path.parent.exists():
                feature_importance_path.parent.mkdir(parents=True, exist_ok=True)

            preds, train_perf, test_perf, eval_key = train_single_fold(
                X_train, X_test, Y_train, Y_test, params.copy(), loss, num_boost_rounds,
                feature_importance_path, features,
            )

            best_test_round = np.argmin(test_perf)
            fold_perfs.append(test_perf[best_test_round])
            fold_df = pd.DataFrame({
                "predicted": preds,
                "actual": Y_test,
                "official_id": test_ids,
                "trial": trial,
                "fold": fold_no
            })
            fold_results.append(fold_df)

            convergence_data.append(pd.DataFrame({
                "train": train_perf,
                "test": test_perf,
                "round": np.arange(len(train_perf)),
                "fold": fold_no,
                "trial": trial
            }))

        mean_perf = np.mean(fold_perfs)
        std_perf = np.std(fold_perfs)
        all_trial_results.append({
            "trial": trial,
            f"test_{eval_key}_mean": mean_perf,
            f"test_{eval_key}_stdev": std_perf,
            **params
        })

        all_predictions.extend(fold_results)
        all_convergence_data.extend(convergence_data)

        if trial % 10 == 0 and trial > 0:
            print(f"Completed {trial} trials")

    result_df = pd.DataFrame(all_trial_results)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_convergence_df = pd.concat(all_convergence_data, ignore_index=True)

    results_fname = f"{set_name}_{eval_key}_all_preds.csv"
    all_predictions_df.to_csv(Path(results_folder) / results_fname, index=False)

    overall_mean = result_df[f"test_{eval_key}_mean"].mean()
    overall_std = result_df[f"test_{eval_key}_mean"].std()
    print(f"Performance across all trials: {overall_mean:.4f} ± {overall_std:.4f} ({eval_key})")

    return result_df, all_predictions_df, all_convergence_df
