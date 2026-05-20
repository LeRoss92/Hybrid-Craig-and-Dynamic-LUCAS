import numpy as np
import warnings
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 warnings
import matplotlib.pyplot as plt
# Suppress numpy division warnings in correlation calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
# Also suppress warnings from numpy's corrcoef function
np.seterr(divide='ignore', invalid='ignore')
# Suppress sklearn Parallel/delayed UserWarning from GridSearchCV (sklearn uses delayed with joblib.Parallel internally)
warnings.filterwarnings('ignore', category=UserWarning, message='.*sklearn.utils.parallel.delayed.*')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.base import clone
from sklearn.utils.parallel import Parallel, delayed
import importlib.util

from models import get_models

# from config import get_onehot_cols_for_config


def normalize_config(c):
    """Convert config with predictors/log_predictors lists to {predictors: {normal, log}, onehot_prefixes}."""
    if isinstance(c.get('predictors'), dict):
        return c
    preds = c.get('predictors') or []
    log_preds = c.get('log_predictors') or []
    return {
        **c,
        'predictors': {"normal": tuple(preds), "log": tuple(log_preds)},
        'onehot_prefixes': c.get('onehot_prefixes', c.get('categoricals', [])),
    }


def get_array(df, col):
    arr = np.array(df[col]) # df column -> np
    try: return arr.astype(float) # return if floats
    except: return LabelEncoder().fit_transform(arr.astype(str)) # encode strings of categoricals

def kge(y_true, y_pred, sub=False):
    # Handle NaN/inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:  # Need at least 2 valid points for correlation
        return (np.nan, np.nan, np.nan, np.nan) if sub else np.nan
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Check for zero variance before correlation calculation
    std_true = np.std(y_true_clean)
    std_pred = np.std(y_pred_clean)
    
    # If either has zero variance, correlation is undefined
    if std_true == 0 or std_pred == 0 or not np.isfinite(std_true) or not np.isfinite(std_pred):
        r = 0.0
    else:
        # Suppress warnings from np.corrcoef when it encounters edge cases
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                corr_matrix = np.corrcoef(y_true_clean, y_pred_clean)
                r = corr_matrix[0, 1] if np.isfinite(corr_matrix[0, 1]) else 0.0
            except:
                r = 0.0
    
    # Handle division by zero for std
    alpha = std_pred / std_true if std_true > 0 and np.isfinite(std_true) else 1.0  # Default to 1.0 if zero variance
    
    # Handle division by zero for mean
    mean_true = np.mean(y_true_clean)
    mean_pred = np.mean(y_pred_clean)
    beta = mean_pred / mean_true if mean_true != 0 and np.isfinite(mean_true) else 1.0  # Default to 1.0 if zero mean
    
    # Calculate KGE, handling NaN/inf
    if not (np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta)):
        return (np.nan, r, alpha, beta) if sub else np.nan
    
    KGE = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    return (KGE, r, alpha, beta) if sub else KGE

def prepare_data(df, target, predictors):
    y = get_array(df, target) # get target (incl. nans)
    X = np.column_stack([get_array(df, col) for col in predictors]) # get all predictors (incl. nan)
    valid = ~np.isnan(y) & ~np.all(np.isnan(X), axis=1) # identify rows with: target + at least 1 predictor
    return X[valid], y[valid], valid # filter + mask of values to keep

def plot_pred_tar_dist(target_name, feature_names, y_big_train, X_big_train):
    n_total = len(feature_names) + 1  # +1 for target
    n_cols = 10
    n_rows = (n_total + n_cols - 1) // n_cols

    # Make subplot smaller by reducing figsize:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 1.6))
    fig.suptitle('X/y distributions', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    # Plot target distribution with red bins
    axes[0].hist(y_big_train, bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[0].set_title(target_name, fontsize=9)
    axes[0].tick_params(labelsize=7)

    # Plot predictor distributions
    for i, feature_name in enumerate(feature_names):
        ax = axes[i + 1]
        values = X_big_train[:, i]
        ax.hist(values[~np.isnan(values)], bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(feature_name, fontsize=9)
        ax.tick_params(labelsize=7)

    # Remove empty subplots
    for j in range(n_total, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f'5_results/5_{target_name}_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

def _eval_cand(c, sel, Xp, yp, mc, cfg, scorer):
    """Evaluate adding a single feature index c to selection sel."""
    fs = sel + [c]
    n_jobs_gs = mc.get('n_jobs', 1)
    s = GridSearchCV(mc['model'], mc['params'], cv=cfg['n_folds_HP_opt'], scoring=scorer, n_jobs=n_jobs_gs, verbose=0)
    s.fit(Xp[:, fs], yp)
    return (c, s.best_score_, s)

def _eval_cand_group(indices, sel, Xp, yp, mc, cfg, scorer):
    """Evaluate adding a whole group (all indices) to selection sel."""
    fs = sel + indices
    n_jobs_gs = mc.get('n_jobs', 1)
    s = GridSearchCV(mc['model'], mc['params'], cv=cfg['n_folds_HP_opt'], scoring=scorer, n_jobs=n_jobs_gs, verbose=0)
    s.fit(Xp[:, fs], yp)
    return (tuple(indices), s.best_score_, s)

def process_fold(fold_idx, X_train_all, y_train_all, split_train_labels, X_test, y_test, target_name, feature_names, config, all_MODELS, fold_values, feature_groups=None):
    """Process a single fold - can be called in parallel.
    feature_groups: dict mapping group_id (str) -> list of feature indices. These are selected all-or-nothing.
    """
    # Set matplotlib backend in worker process to avoid X11 warnings
    import matplotlib
    matplotlib.use('Agg', force=True)
    # Suppress sklearn Parallel/delayed UserWarning (raised by GridSearchCV when n_jobs>1 in worker processes)
    warnings.filterwarnings('ignore', category=UserWarning, message='.*sklearn.utils.parallel.delayed.*')
    # Pin each fold worker to one visible GPU for multi-GPU parallel runs.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_gpus = [g.strip() for g in visible.split(",") if g.strip() != ""]
    if visible_gpus:
        selected_gpu = visible_gpus[int(fold_idx) % len(visible_gpus)]
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
    else:
        selected_gpu = "all-visible"
    
    val_mask = split_train_labels == fold_idx # get respective fold indices (test)
    train_mask = ~val_mask # everything else valid is train
    X_train, X_val, y_train, y_val = X_train_all[train_mask], X_train_all[val_mask], y_train_all[train_mask], y_train_all[val_mask] # get train and val X and y
    y_train_orig, y_val_orig = y_train.copy(), y_val.copy() # Keep original y for KGE calculation
    if config.get('target_log', False): # Apply log transform to target
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
    # -------------------- Preprocessing --------------------
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]) # nans in predictor to median + z-scoring
    X_train_p = preprocessor.fit_transform(X_train) # apply preprocessing to train 
    X_val_p   = preprocessor.transform(X_val) # use the train statistics to also preprocess val
    target_scaler = StandardScaler() # Always scale target for model training
    y_train_p = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_p = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    if fold_idx == fold_values[0]:
        plot_pred_tar_dist(target_name, feature_names, y_train_p, X_train_p)
    print(f"TARGET: {target_name} | Train={len(X_train)} Val={len(X_val)} | Test={len(X_test)} | Fold {fold_idx}")
    
    feature_groups = feature_groups or {}
    all_group_indices = set()
    for idx_list in feature_groups.values():
        all_group_indices.update(idx_list)
    remaining_singles = [i for i in range(X_train_p.shape[1]) if i not in all_group_indices]
    remaining_groups = list(feature_groups.items())  # [(group_id, [indices]), ...]

    fold_results = []
    for model_type in config['models']:
            selected_idx = []
            remaining_s = list(remaining_singles)
            remaining_g = list(remaining_groups)
            model_config = all_MODELS[model_type]
            kge_scorer = make_scorer(kge, greater_is_better=True)
            best_overall_score, best_overall_model, best_overall_features, best_overall_hyperparams = -np.inf, None, None, None
            # XGB uses GPU per fold; for LinReg/Piecewise allow candidate parallelism (2TB node)
            n_jobs_c = 1 if model_type == 'XGB' else config.get('n_jobs_candidates', 6)
            n_candidates = len(remaining_s) + len(remaining_g)
            for _ in range(min(config['max_features'], n_candidates)):
                cand_results = []
                if n_jobs_c > 1:
                    cand_results.extend(Parallel(n_jobs=n_jobs_c, backend='loky')(delayed(_eval_cand)(c, selected_idx, X_train_p, y_train_p, model_config, config, kge_scorer) for c in remaining_s))
                    cand_results.extend(Parallel(n_jobs=n_jobs_c, backend='loky')(delayed(_eval_cand_group)(idxs, selected_idx, X_train_p, y_train_p, model_config, config, kge_scorer) for _, idxs in remaining_g))
                else:
                    cand_results.extend([_eval_cand(c, selected_idx, X_train_p, y_train_p, model_config, config, kge_scorer) for c in remaining_s])
                    cand_results.extend([_eval_cand_group(idxs, selected_idx, X_train_p, y_train_p, model_config, config, kge_scorer) for _, idxs in remaining_g])
                if not cand_results:
                    break
                best_result = max(cand_results, key=lambda r: r[1])
                best_cand, best_score, best_search = best_result
                if isinstance(best_cand, tuple):
                    selected_idx.extend(best_cand)
                    best_idxs_set = set(best_cand)
                    remaining_g = [(gid, idxs) for gid, idxs in remaining_g if set(idxs) != best_idxs_set]
                else:
                    selected_idx.append(best_cand)
                    remaining_s.remove(best_cand)
                if best_score > best_overall_score:
                    best_overall_score, best_overall_features, best_overall_hyperparams, best_overall_model = best_score, selected_idx.copy(), best_search.best_params_, best_search.best_estimator_
            # -------------------- Train best model and evaluate --------------------
            final_model = clone(best_overall_model)
            final_model.fit(X_train_p[:, best_overall_features], y_train_p)
            X_test_p = preprocessor.transform(X_test)
            y_train_pred = target_scaler.inverse_transform(final_model.predict(X_train_p[:, best_overall_features]).reshape(-1, 1)).flatten()
            y_val_pred = target_scaler.inverse_transform(final_model.predict(X_val_p[:, best_overall_features]).reshape(-1, 1)).flatten()
            y_test_pred = target_scaler.inverse_transform(final_model.predict(X_test_p[:, best_overall_features]).reshape(-1, 1)).flatten()
            if config.get('target_log', False):
                y_train_pred, y_val_pred, y_test_pred = np.expm1(y_train_pred), np.expm1(y_val_pred), np.expm1(y_test_pred)
            train_KGE, val_KGE, test_KGE = [kge(y_true, y_pred) for y_true, y_pred in [(y_train_orig, y_train_pred), (y_val_orig, y_val_pred), (y_test, y_test_pred)]]
            train_R2, val_R2, test_R2 = [r2_score(y_true, y_pred) for y_true, y_pred in [(y_train_orig, y_train_pred), (y_val_orig, y_val_pred), (y_test, y_test_pred)]]
            train_RMSE, val_RMSE, test_RMSE = [np.sqrt(mean_squared_error(y_true, y_pred)) for y_true, y_pred in [(y_train_orig, y_train_pred), (y_val_orig, y_val_pred), (y_test, y_test_pred)]]
            fold_results.append({
                'target': target_name,
                'fold': fold_idx,
                'model_type': model_type,
                'hyperparams': str(best_overall_hyperparams),
                'model': final_model,
                'preprocessor': preprocessor,
                'target_scaler': target_scaler,
                'features': [feature_names[i] for i in best_overall_features],
                'feature_indices': best_overall_features,
                'train_KGE': train_KGE,
                'train_R2': train_R2,
                'train_RMSE': train_RMSE,
                'val_KGE': val_KGE,
                'val_R2': val_R2,
                'val_RMSE': val_RMSE,
                'test_KGE': test_KGE,
                'test_R2': test_R2,
                'test_RMSE': test_RMSE
            })
    return fold_results

def find(df, config, split_col="split", results_suffix=None, fold_filter=None):
    results = pd.DataFrame()
    target_name = config["target_name"]
    all_MODELS = get_models(config['seed'])
    feature_names = config['predictors']["normal"] + config['predictors']["log"]
    onehot_cols = []
    feature_groups = {}
    if get_onehot_cols_for_config and config.get('onehot_prefixes'):
        onehot_cols, onehot_specs = get_onehot_cols_for_config(df, config)
        base_len = len(config['predictors']["normal"]) + len(config['predictors']["log"])
        for prefix in config['onehot_prefixes']:
            if prefix in onehot_specs:
                start = base_len + sum(len(onehot_specs[p]) for p in config['onehot_prefixes'] if config['onehot_prefixes'].index(p) < config['onehot_prefixes'].index(prefix))
                feature_groups[prefix] = list(range(start, start + len(onehot_specs[prefix])))
        feature_names = list(feature_names) + onehot_cols
    X_raw, y, valid_mask = prepare_data(df, target_name, feature_names)
    n_normal = len(config['predictors']["normal"])
    n_log = len(config['predictors']["log"])
    X_raw[:, n_normal:n_normal + n_log] = np.log1p(X_raw[:, n_normal:n_normal + n_log])
    split_labels = np.array(df.loc[valid_mask, split_col])
    test_mask = split_labels == 'test'
    X_test, y_test = X_raw[test_mask], y[test_mask]
    X_train_all, y_train_all = X_raw[~test_mask], y[~test_mask]
    split_train_labels = split_labels[~test_mask]
    fold_values = sorted({int(v) for v in pd.unique(split_train_labels) if v != 'test'})
    if fold_filter is not None:
        if fold_filter not in fold_values:
            raise ValueError(f"fold_filter={fold_filter} not in fold_values {fold_values}")
        fold_values = [fold_filter]

    n_jobs_folds = config.get('n_jobs_folds', 1)
    if n_jobs_folds > 1:
        fold_results_list = Parallel(n_jobs=n_jobs_folds, backend='loky')(
            delayed(process_fold)(fold_idx, X_train_all, y_train_all, split_train_labels, X_test, y_test, target_name, feature_names, config, all_MODELS, fold_values, feature_groups)
            for fold_idx in fold_values
        )
        for fold_results in fold_results_list:
            results = pd.concat([results, pd.DataFrame(fold_results)], ignore_index=True)
    else:
        for fold_idx in fold_values:
            fold_results = process_fold(fold_idx, X_train_all, y_train_all, split_train_labels, X_test, y_test, target_name, feature_names, config, all_MODELS, fold_values, feature_groups)
            results = pd.concat([results, pd.DataFrame(fold_results)], ignore_index=True)
    
    out_name = f'5_results/5_{target_name}'
    if results_suffix is not None:
        out_name = f"{out_name}_{results_suffix}"
    results.to_pickle(f'{out_name}_results.pkl')
    return results

def apply_models(df, results, pred_config, model_year_suffix=None, col_suffix=None, train_config=None):
    """Apply trained models for prediction.
    When doing cross-year prediction (pred_config != training config), pass train_config
    so the feature structure (incl. one-hot) matches what the model was trained on.
    If train_config is None, use pred_config for everything (same-config application).
    """
    pred_columns = {} # to store predictions
    feature_names = pred_config['predictors']['normal'] + pred_config['predictors']['log']
    onehot_config = train_config if train_config is not None else pred_config
    onehot_cols = []
    if get_onehot_cols_for_config and onehot_config.get('onehot_prefixes'):
        onehot_cols, _ = get_onehot_cols_for_config(df, onehot_config)
        feature_names = list(feature_names) + onehot_cols
    n_normal = len(pred_config['predictors']['normal'])
    n_log = len(pred_config['predictors']['log'])
    for fold_idx in results['fold'].unique(): # one prediction per fold (uncertainty)
        fold_results = results[results['fold'] == fold_idx] # filter to have evaluation results only of that fold
        best_model_row = fold_results.loc[fold_results['val_KGE'].idxmax()] # identify best model (via val, not test!)
        model = best_model_row['model'] # get model, preprocessors and indices of the features from selection
        preprocessor = best_model_row['preprocessor']
        target_scaler = best_model_row['target_scaler']
        feature_indices = best_model_row['feature_indices']
        X_raw = np.column_stack([get_array(df, col) for col in feature_names])
        X_raw[:, n_normal:n_normal + n_log] = np.log1p(X_raw[:, n_normal:n_normal + n_log])
        X_p = preprocessor.transform(X_raw) # impute and zscore (with statistics of model training)
        y_pred_scaled = model.predict(X_p[:, feature_indices]) # predict
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten() # re-scale predictions to original scale
        if pred_config['target_log']: # re-log if target was logged
            y_pred = np.expm1(y_pred)
        base = pred_config['target_name']
        if col_suffix is not None:
            base = f"{base}_{col_suffix}"
        col_name = f"{base}_pred_{model_year_suffix}_{fold_idx}" if model_year_suffix is not None else f"{base}_pred_{fold_idx}"
        pred_columns[col_name] = y_pred # store predictions
    return pd.concat([df, pd.DataFrame(pred_columns, index=df.index)], axis=1) # add predictions

def haversine(lat1, lon1, lat2, lon2):
    """in meters"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) # convert decimal degrees to radians
    dlat = lat2 - lat1 
    dlon = lon2 - lon1
    # haversine formula
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of earth in meters
    return c * r

def apply_models_per_method(df, results, pred_config, fold_idx):
    """Add n_methods * n_prediction_years columns. One prediction column per (model_type, target)."""
    feature_names = pred_config['predictors']['normal'] + pred_config['predictors']['log']
    onehot_cols = []
    if get_onehot_cols_for_config and pred_config.get('onehot_prefixes'):
        onehot_cols, _ = get_onehot_cols_for_config(df, pred_config)
        feature_names = list(feature_names) + onehot_cols
    n_normal = len(pred_config['predictors']['normal'])
    n_log = len(pred_config['predictors']['log'])

    fold_results = results[results["fold"] == fold_idx]
    pred_columns = {}
    for _, row in fold_results.iterrows():
        model_type = row["model_type"]
        model = row["model"]
        preprocessor = row["preprocessor"]
        target_scaler = row["target_scaler"]
        feature_indices = row["feature_indices"]

        X_raw = np.column_stack([get_array(df, col) for col in feature_names])
        X_raw[:, n_normal:n_normal + n_log] = np.log1p(X_raw[:, n_normal:n_normal + n_log])
        n_expected = preprocessor.named_steps['imputer'].n_features_in_
        n_actual = X_raw.shape[1]
        if n_expected != n_actual:
            print(f"FEATURE MISMATCH: preprocessor expects {n_expected}, got {n_actual}")
            print(f"  target: {pred_config['target_name']} | model: {model_type}")
            print(f"  onehot_prefixes: {pred_config.get('onehot_prefixes', [])}")
            print(f"  feature_names ({n_actual}): {feature_names}")
        X_p = preprocessor.transform(X_raw)
        y_pred_scaled = model.predict(X_p[:, feature_indices])
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        if pred_config["target_log"]:
            y_pred = np.expm1(y_pred)

        col_name = f"{pred_config['target_name']}_{model_type}"
        pred_columns[col_name] = y_pred

    return pd.concat([df, pd.DataFrame(pred_columns, index=df.index)], axis=1)


def compute_pred_median(df, prefix, years, cross_pairs=None, col_suffix=None):
    median_cols = {}
    for year in years:
        year_key = f"{year}_{col_suffix}" if col_suffix is not None else year
        col_base = f"{prefix}{year_key}_pred_"
        pred_cols = [col for col in df.columns if col.startswith(col_base) and col[len(col_base):].isdigit()]
        pred_cols.sort(key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)
        median_cols[f"{prefix}{year_key}_pred_median"] = df[pred_cols].median(axis=1, skipna=True)
    if cross_pairs:
        for target_year, model_year in cross_pairs:
            year_key = f"{target_year}_{col_suffix}" if col_suffix is not None else target_year
            col_base = f"{prefix}{year_key}_pred_{model_year}_"
            pred_cols = [col for col in df.columns if col.startswith(col_base) and col[len(col_base):].isdigit()]
            pred_cols.sort(key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)
            median_cols[f"{prefix}{year_key}_pred_{model_year}_median"] = df[pred_cols].median(axis=1, skipna=True)
    df = pd.concat([df, pd.DataFrame(median_cols, index=df.index)], axis=1)
    return df

from functools import partial

import jax
import jax.numpy as jnp
import diffrax as dfx


def init_mlp(key, sizes):
    """
    Initialize parameters for a multilayer perceptron (MLP).

    Args:
        key: jax.random.PRNGKey
            Random key for parameter initialization.
        sizes: list or tuple of int
            A list of sizes (number of units) for each layer, 
            where len(sizes) = number of layers + 1.
            For example, [in_dim, hidden1, hidden2, ..., out_dim].

    Returns:
        params: list of tuples
            A list where each element is a tuple (w, b) representing the weights and biases 
            for a layer. 
            - w has shape (in_dim, out_dim)
            - b has shape (out_dim,)
            The weights for the last layer are scaled differently (0.01), 
            all other layers use He initialization (sqrt(2.0/in_dim)).
    
    Example:
        params = init_mlp(jax.random.PRNGKey(0), [4, 64, 32, 1])
    """
    params = []
    keys = jax.random.split(key, len(sizes))
    last_idx = len(sizes) - 2
    for i, (k, (in_dim, out_dim)) in enumerate(zip(keys, zip(sizes[:-1], sizes[1:]))):
        if i == last_idx:
            w = jax.random.normal(k, (in_dim, out_dim)) * 0.01
        else:
            w = jax.random.normal(k, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
        b = jnp.zeros((out_dim,))
        params.append((w, b))
    return params


def mlp_forward(params, x):
    """
    Forward pass through a multilayer perceptron (MLP).

    Args:
        params: list of tuples
            A list of (w, b) tuples for each layer of the MLP,
            where w is the weight matrix and b is the bias vector.
        x: jnp.ndarray
            Input tensor of shape (batch_size, input_dim) or (input_dim,).

    Returns:
        jnp.ndarray:
            The output of the MLP. For a final layer with out_dim units, shape will be (batch_size, out_dim).

    Notes:
        - Uses GELU nonlinearity for all hidden layers.
        - The last layer is linear (no activation function applied).
    """
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def constrain_to_range(raw, mins, maxs):
    return mins + (maxs - mins) * jax.nn.sigmoid(raw)


def normalize_targets(y, target_mean, target_std):
    return (y - target_mean) / target_std


def pools_to_loss_targets(y_cmp, y0, use_dynamic, targets_arg):
    """
    Converts model predictions of compartment pools into loss targets appropriate for different training tasks.

    Args:
        y_cmp (jnp.ndarray): Model output pools representing either the change in pool sizes (if use_dynamic=True) 
            or the absolute pool sizes (if use_dynamic=False). Shape (..., 3).
        y0 (jnp.ndarray): Initial pool values (required if use_dynamic=True), same shape as y_cmp.
        use_dynamic (bool): 
            - If True, y_cmp contains the *change* in pool sizes since y0, and final total pools are y_cmp + y0.
            - If False, y_cmp contains the *absolute* pool sizes, and y0 is ignored.
        targets_arg (str): Specifies which target variables to compute. Options are:
            - "SOC": total SOC only.
            - "SOC,MICi": total SOC and (dynamic: ΔCb | steady: microbial fraction).
            - "SOC,MAOCi": total SOC and (dynamic: ΔCm | steady: MAOC fraction).
            - "SOC,MAOCi,MICi": total SOC and both MAOC and MIC (dynamic: ΔCm, ΔCb | steady: fractions).

    Returns:
        jnp.ndarray: Target array for loss computation, concatenated as needed for the output type.
            - Shape (..., n_targets), where n_targets depends on targets_arg.

    Raises:
        KeyError: If targets_arg is not a recognized target specification.

    Notes:
        - The order of pools in y_cmp and y0 is assumed to be: [Cp, Cb, Cm] (particulate, microbial, MAOC).
        - Steady-state: MICi/MAOCi are fractions of final pools. Dynamic: MICi/MAOCi are ΔCb / ΔCm.
    """
    soc_sum = jnp.sum(y_cmp, axis=-1, keepdims=True)
    if targets_arg == "SOC":
        return soc_sum
    if use_dynamic:
        mic_delta = y_cmp[:, 1:2]
        maoc_delta = y_cmp[:, 2:3]
        if targets_arg == "SOC,MICi":
            return jnp.concatenate([soc_sum, mic_delta], axis=-1)
        if targets_arg == "SOC,MAOCi":
            return jnp.concatenate([soc_sum, maoc_delta], axis=-1)
        if targets_arg == "SOC,MAOCi,MICi":
            return jnp.concatenate([soc_sum, maoc_delta, mic_delta], axis=-1)
        raise KeyError(targets_arg)
    y_fin = y_cmp
    soc_lvl = jnp.sum(y_fin, axis=-1, keepdims=True) + 1e-12
    mic_r = y_fin[:, 1:2] / soc_lvl
    maoc_r = y_fin[:, 2:3] / soc_lvl
    if targets_arg == "SOC,MICi":
        return jnp.concatenate([soc_sum, mic_r], axis=-1)
    if targets_arg == "SOC,MAOCi":
        return jnp.concatenate([soc_sum, maoc_r], axis=-1)
    if targets_arg == "SOC,MAOCi,MICi":
        return jnp.concatenate([soc_sum, maoc_r, mic_r], axis=-1)
    raise KeyError(targets_arg)


def build_param_matrix(
    net_params,
    global_raw,
    x_batch,
    npp_I_batch,
    *,
    param_mins,
    param_maxs,
    global_mask,
):
    raw_local = jax.vmap(lambda x: mlp_forward(net_params, x))(x_batch)
    local_params = constrain_to_range(raw_local, param_mins, param_maxs)
    if global_raw is not None:
        global_params = constrain_to_range(global_raw, param_mins, param_maxs)
        local_params = jnp.where(global_mask, global_params, local_params)
    local_params = local_params.at[:, 0].set(npp_I_batch)
    return local_params


def eval_components(params, x_batch, npp_I_batch, y0_batch, y_target, *, param_mins, param_maxs, global_mask, use_dynamic, batched_sim, batched_steady, target_mean, target_std, targets_arg):
    p_pred = build_param_matrix(params["net"], params["global"], x_batch, npp_I_batch, param_mins=param_mins, param_maxs=param_maxs, global_mask=global_mask)
    if use_dynamic:
        y_pred = batched_sim(p_pred, y0_batch)
        y_pred_compare = y_pred - y0_batch
    else:
        y_pred_compare = batched_steady(p_pred)
    y_pred_derived = pools_to_loss_targets(y_pred_compare, y0_batch, use_dynamic, targets_arg)
    y_pred_norm = normalize_targets(y_pred_derived, target_mean, target_std)
    y_target_norm = normalize_targets(y_target, target_mean, target_std)
    diff = y_pred_norm - y_target_norm
    delta = 0.5
    abs_diff = jnp.abs(diff)
    huber = jnp.where(abs_diff <= delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    return jnp.mean(huber, axis=0)


def eval_loss(
    params,
    x_batch,
    npp_I_batch,
    y0_batch,
    y_target,
    weights,
    *,
    param_mins,
    param_maxs,
    global_mask,
    use_dynamic,
    batched_sim,
    batched_steady,
    target_mean,
    target_std,
    targets_arg,
    ):
    per_component = eval_components(
        params,
        x_batch,
        npp_I_batch,
        y0_batch,
        y_target,
        param_mins=param_mins,
        param_maxs=param_maxs,
        global_mask=global_mask,
        use_dynamic=use_dynamic,
        batched_sim=batched_sim,
        batched_steady=batched_steady,
        target_mean=target_mean,
        target_std=target_std,
        targets_arg=targets_arg,
    )
    weights = weights / (jnp.sum(weights) + 1e-8)
    loss = jnp.sum(per_component * weights)
    return loss, per_component


def init_adam(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return m, v


def clip_by_global_norm(grads, max_norm):
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(jnp.sum(jnp.array([jnp.sum(g ** 2) for g in leaves])))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def l2_norm(params):
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.sum(jnp.array([jnp.sum(p ** 2) for p in leaves]))


def eval_r2(
    params,
    x_batch,
    npp_I_batch,
    y0_batch,
    y_target,
    *,
    param_mins,
    param_maxs,
    global_mask,
    use_dynamic,
    batched_sim,
    batched_steady,
    targets_arg,
):
    p_pred = build_param_matrix(
        params["net"],
        params["global"],
        x_batch,
        npp_I_batch,
        param_mins=param_mins,
        param_maxs=param_maxs,
        global_mask=global_mask,
    )
    if use_dynamic:
        y_pred = batched_sim(p_pred, y0_batch)
        y_pred_compare = y_pred - y0_batch
    else:
        y_pred_compare = batched_steady(p_pred)
    y_pred_derived = pools_to_loss_targets(y_pred_compare, y0_batch, use_dynamic, targets_arg)
    ss_res = jnp.sum((y_target - y_pred_derived) ** 2, axis=0)
    ss_tot = jnp.sum((y_target - jnp.mean(y_target, axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@partial(jax.jit, static_argnames=("use_dynamic", "batched_sim", "batched_steady", "targets_arg"))
def train_step(
    params,
    opt_state,
    x_batch,
    npp_I_batch,
    y0_batch,
    y_batch,
    lr_t,
    step,
    weights,
    *,
    param_mins,
    param_maxs,
    global_mask,
    use_dynamic,
    batched_sim,
    batched_steady,
    target_mean,
    target_std,
    targets_arg,
):
    (loss, per_component), grads = jax.value_and_grad(eval_loss, has_aux=True)(
        params,
        x_batch,
        npp_I_batch,
        y0_batch,
        y_batch,
        weights,
        param_mins=param_mins,
        param_maxs=param_maxs,
        global_mask=global_mask,
        use_dynamic=use_dynamic,
        batched_sim=batched_sim,
        batched_steady=batched_steady,
        target_mean=target_mean,
        target_std=target_std,
        targets_arg=targets_arg,
    )
    weight_decay = 1e-4
    loss = loss + weight_decay * l2_norm(params)
    grads = clip_by_global_norm(grads, 1.0)
    m, v = opt_state
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i, m, grads)
    v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1 - beta2) * (g_i ** 2), v, grads)
    m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1 - beta1 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1 - beta2 ** step), v)
    params = jax.tree_util.tree_map(
        lambda p, m_i, v_i: p - lr_t * m_i / (jnp.sqrt(v_i) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, (m, v), loss, per_component


def vector_field(model_fn, t, y, args):
    return model_fn(t, y, args)


def simulate_final_state(params, y0, t0, t1, dt0, term, solver):
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=params,
        saveat=dfx.SaveAt(t1=True),
    )
    return solution.ys[0]
