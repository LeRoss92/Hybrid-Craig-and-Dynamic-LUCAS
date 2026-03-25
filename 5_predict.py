# start over here:
# check if everything is fine with splits
# do only LR on all targets -> print R2 rigth away
# only if that is fine, do other stuff...

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.cross_decomposition')  # PLSR NaN/divide

# load dataset
import time
import numpy as np
import pickle
import pandas as pd
with open("4_with_nc.pkl", "rb") as f:
    df = pickle.load(f)#

from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from config import TARGET_CONFIG, TRAIN_DEFAULTS, get_onehot_cols_for_config

vars_order = ['BD', 'MAOC', 'MIC', 'dSOC_15_18', 'dSOC_09_18', 'SOC09', 'SOC18', 'SOC15']
N_JOBS = 70
from classic_models import get_models
models = get_models()

MAX_FEATURES = TRAIN_DEFAULTS['max_features']
MODEL_NAMES = TRAIN_DEFAULTS['models']

def _make_model(model_config, params_combo, n_features=None):
    """Create model with params. If n_features given, cap n_components for PLSR/PCA-like estimators."""
    m = clone(model_config['model'])
    params = dict(params_combo) if params_combo else {}
    if n_features is not None and 'n_components' in params:
        params['n_components'] = min(params['n_components'], n_features)
    if params:
        m.set_params(**params)
    return m

def forward_select(X_train_p, X_val_p, y_train, y_val, model_config, params_combo, max_features):
    """Additive forward feature selection: iteratively add feature that improves val R² most."""
    n_features = X_train_p.shape[1]
    selected_idx = []
    remaining = list(range(n_features))
    best_val_r2 = -np.inf
    for _ in range(min(max_features, n_features)):
        best_c, best_score = None, -np.inf
        for c in remaining:
            idxs = selected_idx + [c]
            n_cols = len(idxs)
            try:
                m = _make_model(model_config, params_combo, n_features=n_cols)
                m.fit(X_train_p[:, idxs], y_train)
                pred = m.predict(X_val_p[:, idxs])
                if not np.all(np.isfinite(pred)):
                    score = -np.inf
                else:
                    score = r2_score(y_val, pred)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                score = -np.inf
            if score > best_score:
                best_score, best_c = score, c
        if best_c is None or best_score <= best_val_r2:
            break
        best_val_r2 = best_score
        selected_idx.append(best_c)
        remaining.remove(best_c)
    return selected_idx

def process_one_fold(var, fold, X_train, y_train, X_val, y_val, X_test, y_test,
                    train_idx, val_idx, test_idx, n_normal, n_log):
    """Process a single (var, fold): per model, pick best (params, raw/log) by val R²; return metrics + predictions."""
    def to_signed_log(y):
        return np.sign(y) * np.log1p(np.abs(y))
    def from_signed_log(y_log):
        return np.sign(y_log) * np.expm1(np.abs(y_log))

    X_train = np.asarray(X_train, dtype=float, order='C').copy()
    X_val = np.asarray(X_val, dtype=float, order='C').copy()
    X_train[:, n_normal:n_normal + n_log] = np.log1p(X_train[:, n_normal:n_normal + n_log])
    X_val[:, n_normal:n_normal + n_log] = np.log1p(X_val[:, n_normal:n_normal + n_log])
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    y_train_log = to_signed_log(y_train)
    y_val_log = to_signed_log(y_val)

    fold_results = []
    for model_name in MODEL_NAMES:
        model_config = models[model_name]
        param_grid = model_config.get('params', {}) or {}
        best_val_r2 = -np.inf
        best_result = None
        for params_combo in ParameterGrid(param_grid) if param_grid else [{}]:
            try:
                # Option 1: raw target
                selected_raw = forward_select(X_train_p, X_val_p, y_train, y_val, model_config, params_combo, MAX_FEATURES)
                m_raw = _make_model(model_config, params_combo, n_features=len(selected_raw))
                m_raw.fit(X_train_p[:, selected_raw], y_train)
                pred_train_raw = m_raw.predict(X_train_p[:, selected_raw])
                pred_val_raw = m_raw.predict(X_val_p[:, selected_raw])
                pred_test_raw = m_raw.predict(X_test_p[:, selected_raw])
                val_r2_raw = r2_score(y_val, pred_val_raw)
                train_r2_raw = r2_score(y_train, pred_train_raw)
                test_r2_raw = r2_score(y_test, pred_test_raw)
                # Option 2: log target
                selected_log = forward_select(X_train_p, X_val_p, y_train_log, y_val_log, model_config, params_combo, MAX_FEATURES)
                m_log = _make_model(model_config, params_combo, n_features=len(selected_log))
                m_log.fit(X_train_p[:, selected_log], y_train_log)
                pred_train_log = from_signed_log(m_log.predict(X_train_p[:, selected_log]))
                pred_val_log = from_signed_log(m_log.predict(X_val_p[:, selected_log]))
                pred_test_log = from_signed_log(m_log.predict(X_test_p[:, selected_log]))
                val_r2_log = r2_score(y_val, pred_val_log)
                train_r2_log = r2_score(y_train, pred_train_log)
                test_r2_log = r2_score(y_test, pred_test_log)
                # Pick better raw vs log for this model
                if val_r2_log > val_r2_raw:
                    cand = (val_r2_log, train_r2_log, val_r2_log, test_r2_log, params_combo,
                            pred_train_log, pred_val_log, pred_test_log)
                else:
                    cand = (val_r2_raw, train_r2_raw, val_r2_raw, test_r2_raw, params_combo,
                            pred_train_raw, pred_val_raw, pred_test_raw)
                if cand[0] > best_val_r2:
                    best_val_r2 = cand[0]
                    best_result = {
                        'train r2': cand[1], 'val r2': cand[2], 'test r2': cand[3],
                        'params': cand[4],
                        'pred_train': cand[5], 'pred_val': cand[6], 'pred_test': cand[7],
                        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
                    }
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                pass
        if best_result is not None:
            fold_results.append({
                'var': var,
                'fold': fold,
                'model': model_name,
                'train r2': best_result['train r2'],
                'val r2': best_result['val r2'],
                'test r2': best_result['test r2'],
                'params': str(best_result['params']),
                'pred_train': best_result['pred_train'],
                'pred_val': best_result['pred_val'],
                'pred_test': best_result['pred_test'],
                'train_idx': best_result['train_idx'],
                'val_idx': best_result['val_idx'],
                'test_idx': best_result['test_idx'],
            })
    return fold_results

t0 = time.perf_counter()
tasks = []
for var in vars_order:
    config = TARGET_CONFIG[var]
    target_name = config["target_name"]
    onehot_cols, _ = get_onehot_cols_for_config(df, config)
    pred_cols = list(config['predictors']) + list(config['log_predictors']) + onehot_cols
    n_normal = len(config['predictors'])
    n_log = len(config['log_predictors'])
    test = df[(df['split'] == 'test') & (df[target_name].notna())]
    X_test = test[pred_cols].to_numpy(dtype=float)
    y_test = test[target_name].to_numpy(dtype=float)
    X_test[:, n_normal:n_normal + n_log] = np.log1p(X_test[:, n_normal:n_normal + n_log])
    for fold in range(10):
        train = df[(df['split'] != 'test') & (df['split'] != fold) & (df[target_name].notna())]
        val = df[(df['split'] != 'test') & (df['split'] == fold) & (df[target_name].notna())]
        train_idx = train.index.to_numpy()
        val_idx = val.index.to_numpy()
        test_idx = test.index.to_numpy()
        X_train = train[pred_cols].to_numpy(dtype=float)
        y_train = train[target_name].to_numpy(dtype=float)
        X_val = val[pred_cols].to_numpy(dtype=float)
        y_val = val[target_name].to_numpy(dtype=float)
        tasks.append((var, fold, X_train, y_train, X_val, y_val, X_test.copy(), y_test.copy(),
                      train_idx, val_idx, test_idx, n_normal, n_log))

results = Parallel(n_jobs=N_JOBS, backend='loky')(
    delayed(process_one_fold)(*task) for task in tasks
)
# Each fold returns a list of dicts (one per model); flatten
results_flat = [row for fold_results in results for row in fold_results]
for r in results_flat:
    print(f"var {r['var']} fold {r['fold']} {r['model']} train r2: {r['train r2']:.2f}, val r2: {r['val r2']:.2f}, test r2: {r['test r2']:.2f}")

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('pred_train', 'pred_val', 'pred_test', 'train_idx', 'val_idx', 'test_idx')} for r in results_flat])

# Build output df: original df + one column per (var, fold, model) with train/val/test predictions
out_df = df.copy()
for r in results_flat:
    col = f"pred_{r['var']}_fold{r['fold']}_{r['model']}"
    out_df[col] = np.nan
    out_df.loc[r['train_idx'], col] = r['pred_train']
    out_df.loc[r['val_idx'], col] = r['pred_val']
    out_df.loc[r['test_idx'], col] = r['pred_test']
with open("5_with_predictions.pkl", "wb") as f:
    pickle.dump(out_df, f)
print(f"Saved predictions to 5_with_predictions.pkl ({len(out_df)} rows, {len([c for c in out_df.columns if c.startswith('pred_')])} pred columns)")

import matplotlib.pyplot as plt
model_order = [m for m in MODEL_NAMES if m in results_df['model'].values]
combo_order = [f"{v}-{m}" for v in vars_order for m in model_order]
data = [
    results_df[(results_df['var'] == v) & (results_df['model'] == m)]['test r2'].values
    for v in vars_order for m in model_order
]
plt.figure(figsize=(max(10, len(combo_order) * 0.5), 5))
plt.boxplot(data, tick_labels=combo_order)
plt.ylabel('Test R²')
plt.xlabel('Variable–Model')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.ylim(-0.2, 1.0)
plt.yticks(np.arange(-0.2, 1.05, 0.1))
plt.grid(axis='y', which='both', linestyle='--', alpha=0.7)

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/predict.png', dpi=150, bbox_inches='tight')
print(f"Total time: {(time.perf_counter() - t0) / 60:.2f} min")
