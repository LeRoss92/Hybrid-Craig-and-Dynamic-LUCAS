import time
t0 = time.perf_counter()
import numpy as np
import pickle
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

######################################## load ########################################
from config import TARGET_CONFIG, TRAIN_DEFAULTS
from classic_models import get_models
models = get_models()
MAX_FEATURES = TRAIN_DEFAULTS['max_features']
MODEL_NAMES = TRAIN_DEFAULTS['models']
vars_order = TRAIN_DEFAULTS['vars_order']
N_JOBS = TRAIN_DEFAULTS['N_JOBS']
with open("4_with_nc.pkl", "rb") as f:
    df = pickle.load(f)

######################################## define ########################################
def _make_model(model_config, params_combo, n_features=None):
    """Create model with params. If n_features given, cap n_components for PLSR/PCA-like estimators."""
    m = clone(model_config['model']) # clone the model with hyperparameters, so standard remains untouched (not fitted to anything)
    params = dict(params_combo) if params_combo else {} # use hyperparameter combination if given... else use standard hyperparameters
    if n_features is not None and 'n_components' in params: # if n_components is a hyperparameter -> only relevant for PLSR
        params['n_components'] = min(params['n_components'], n_features) # use the ones specified, or if less predictors then n of predictors
    if params: # if hyperparameters to specify 
        m.set_params(**params) # specify
    return m # return model

def forward_select(X_train_p, X_val_p, y_train, y_val, model_config, params_combo, max_features):
    """Additive forward feature selection: iteratively add feature that improves val R² most."""
    n_features = X_train_p.shape[1] # number of possible features
    selected_idx = [] # to store the indices of selected predictors
    remaining = list(range(n_features)) # indices of remaining predictors which could later be selected
    best_val_r2 = -np.inf # start at worst possible R2
    for _ in range(min(max_features, n_features)): # loop over max features (except there are less than defined max)
        best_c, best_score = None, -np.inf # initialize best predictor and corresponding R2
        for c in remaining: # loop over (remaining) possible predictors
            idxs = selected_idx + [c] # formerly selected + possible new predictor
            n_cols = len(idxs) # number of predictors
            try:
                m = _make_model(model_config, params_combo, n_features=n_cols) # create model with respective hyperparameters
                m.fit(X_train_p[:, idxs], y_train) # fit model on train of the predictors (formerly selected + possible new predictor)
                pred = m.predict(X_val_p[:, idxs]) # predict for val
                if not np.all(np.isfinite(pred)): # if any predicted value is corrupt 
                    score = -np.inf # -> worst
                else: # if val prediction worked
                    score = r2_score(y_val, pred) # calculate r2 (with sk_learn)
            except (ValueError, np.linalg.LinAlgError, RuntimeError): # if even fitting didn't work
                score = -np.inf # -> worst
            if score > best_score: # if val R2 better than any before
                best_score, best_c = score, c # select/overwrite
        if best_c is None or best_score <= best_val_r2: # if no possible additional predictor makes predictions better:
            break # don't select further predictor: less predictors than max
        best_val_r2 = best_score # but if additional predictor helped: new best score
        selected_idx.append(best_c) # but if additional predictor helped: add best new predictor to selection
        remaining.remove(best_c) # remove the selected from the ones to test in next round
    return selected_idx # return indices of selected predictors

def process_one_fold(var, fold, X_train, y_train, X_val, y_val, X_test, y_test,
                    train_idx, val_idx, test_idx, n_normal, n_log, pred_cols):
    """Process a single (var, fold): per model, pick best (params, raw/log) by val R²; return metrics + predictions."""
    def to_signed_log(y): # log the abs value and give it sign: log of negative (changes) makes problems otherwise
        return np.sign(y) * np.log1p(np.abs(y))
    def from_signed_log(y_log): # function to revert (when prediction)
        return np.sign(y_log) * np.expm1(np.abs(y_log))

    X_train = np.asarray(X_train, dtype=float, order='C').copy() # order='C' -> row major
    X_val = np.asarray(X_val, dtype=float, order='C').copy() # rearrange array
    X_train[:, n_normal:n_normal + n_log] = np.log1p(X_train[:, n_normal:n_normal + n_log]) # log respective predictors
    X_val[:, n_normal:n_normal + n_log] = np.log1p(X_val[:, n_normal:n_normal + n_log]) # log respective predictors
    preprocessor = Pipeline([ 
        ("imputer", SimpleImputer(strategy="median")), # implace empty predictors with median
        ("scaler", StandardScaler()) # z-score
    ])
    X_train_p = preprocessor.fit_transform(X_train) # get median (for imputation), mean and var (for z-score) on train
    X_val_p = preprocessor.transform(X_val) # apply with train stats on val predictors
    X_test_p = preprocessor.transform(X_test) # apply with train stats on test predictors
    y_train_log = to_signed_log(y_train) # log train target (if negative special: vide function)
    y_val_log = to_signed_log(y_val) # log val target (if negative special: vide function)

    fold_results = []
    for model_name in MODEL_NAMES: # loop over methods (LR, PLR, XGB...)
        model_config = models[model_name] # get info on model
        param_grid = model_config.get('params', {}) or {} # create latin hypercube of hyperparameter combinations (if HP opt...)
        best_val_r2 = -np.inf # init best metric (worst R2 in -inf)
        best_result = None
        for params_combo in ParameterGrid(param_grid) if param_grid else [{}]: # loop over hyperparameter combos
            try:
                # Option 1: raw target
                selected_raw = forward_select(X_train_p, X_val_p, y_train, y_val, model_config, params_combo, MAX_FEATURES) # select predictors with normal target: indices of selected returned
                m_raw = _make_model(model_config, params_combo, n_features=len(selected_raw)) # get the model (seems to make sense)
                m_raw.fit(X_train_p[:, selected_raw], y_train) # fit the model to selected predictors (on train only)
                pred_train_raw = m_raw.predict(X_train_p[:, selected_raw]) # predict train
                pred_val_raw = m_raw.predict(X_val_p[:, selected_raw]) # predict val
                pred_test_raw = m_raw.predict(X_test_p[:, selected_raw]) # predict test
                val_r2_raw = r2_score(y_val, pred_val_raw) # calculate val R2
                train_r2_raw = r2_score(y_train, pred_train_raw) # calculate train R2
                test_r2_raw = r2_score(y_test, pred_test_raw) # calculate test R2
                # Option 2: log target -> do same with logged target
                selected_log = forward_select(X_train_p, X_val_p, y_train_log, y_val_log, model_config, params_combo, MAX_FEATURES) # select predictors with logged target: indices of selected returned
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
                            pred_train_log, pred_val_log, pred_test_log, selected_log, True)
                else:
                    cand = (val_r2_raw, train_r2_raw, val_r2_raw, test_r2_raw, params_combo,
                            pred_train_raw, pred_val_raw, pred_test_raw, selected_raw, False)
                if cand[0] > best_val_r2: # if val R2 is best (like best hyperparameter combination, log/no log)
                    best_val_r2 = cand[0]
                    best_result = {
                        'train r2': cand[1], 'val r2': cand[2], 'test r2': cand[3],
                        'params': cand[4],
                        'pred_train': cand[5], 'pred_val': cand[6], 'pred_test': cand[7],
                        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
                        'selected': cand[-2], 'target_logged': cand[-1],
                    } # save this results for that model + fold
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                pass # skip if this hyperparameter combo crashed
        if best_result is not None: # if any haperparameter combo worked:
            sel = best_result['selected']
            fold_results.append({
                'var': var,
                'fold': fold,
                'model': model_name,
                'train r2': best_result['train r2'],
                'val r2': best_result['val r2'],
                'test r2': best_result['test r2'],
                'params': best_result['params'],
                'target_logged': best_result['target_logged'],
                'selected_predictors': [pred_cols[i] for i in sel],
                'pred_train': best_result['pred_train'],
                'pred_val': best_result['pred_val'],
                'pred_test': best_result['pred_test'],
                'train_idx': best_result['train_idx'],
                'val_idx': best_result['val_idx'],
                'test_idx': best_result['test_idx'],
                'selected': sel,
            }) # save results (per model/fold)
    return fold_results

def get_onehot_cols_for_config(df, config):
    """Resolve categorical columns from df. config uses 'categoricals' or 'onehot_prefixes'."""
    prefixes = config.get('categoricals', config.get('onehot_prefixes', [])) # list with prefixes of all desired 1-hot columns
    cols = []
    specs = {}
    for p in prefixes: # loop over prefixes/categoricals
        group_cols = [c for c in df.columns if c.startswith(f'{p}_')] # get the columns which have respective prefix
        if group_cols: # if any found
            cols.extend(group_cols) # add columns (data)
            specs[p] = group_cols # add column names
    return cols, specs # return data and column names

######################################## create parallel tasks ########################################
tasks = []
for var in vars_order:
    config = TARGET_CONFIG[var] # info (targets, predictors, inference, log...)
    target_name = config["target_name"] # target name
    onehot_cols, _ = get_onehot_cols_for_config(df, config) # get data of 1-hot variables
    pred_cols = list(config['predictors']) + list(config['log_predictors']) + onehot_cols # all potential predictors together (1-hot might get selected individually)
    n_normal = len(config['predictors']) # number of normal distributed predictors
    n_log = len(config['log_predictors']) # numer of log normal distributed predictors
    test = df[(df['split'] == 'test') & (df[target_name].notna())] # test are all rows where: test + actual target data
    X_test = test[pred_cols].to_numpy(dtype=float) # get test X as array
    y_test = test[target_name].to_numpy(dtype=float) # get test y as array
    X_test[:, n_normal:n_normal + n_log] = np.log1p(X_test[:, n_normal:n_normal + n_log]) # log respective predictors (looks correct) 
    for fold in range(10): # loop over folds
        train = df[(df['split'] != 'test') & (df['split'] != fold) & (df[target_name].notna())] # train is where: not test + not val + actual data
        val = df[(df['split'] != 'test') & (df['split'] == fold) & (df[target_name].notna())] # val is where: not test + val + actual data
        train_idx = train.index.to_numpy() # save train indices
        val_idx = val.index.to_numpy() # save val indices
        test_idx = test.index.to_numpy() # save test indices
        X_train = train[pred_cols].to_numpy(dtype=float) # make train X array
        y_train = train[target_name].to_numpy(dtype=float) # make train y array
        X_val = val[pred_cols].to_numpy(dtype=float) # make val X array
        y_val = val[target_name].to_numpy(dtype=float) # make val y array
        tasks.append((var, fold, X_train, y_train, X_val, y_val, X_test.copy(), y_test.copy(),
                      train_idx, val_idx, test_idx, n_normal, n_log, pred_cols)) # one task pre variable and fold (n_vars*n_folds=n_tasks)

######################################## compute ########################################
results = Parallel(n_jobs=N_JOBS, backend='loky')(delayed(process_one_fold)(*task) for task in tasks)

######################################## post processing ########################################
# Each fold returns a list of dicts (one per model); flatten
results_flat = [row for fold_results in results for row in fold_results]
for r in results_flat:
    print(f"var {r['var']} fold {r['fold']} {r['model']} train r2: {r['train r2']:.2f}, val r2: {r['val r2']:.2f}, test r2: {r['test r2']:.2f}, "
          f"log_target={r['target_logged']}, predictors={r['selected_predictors']}")

# One row per (var, fold, model): hyperparameters dict, log-target flag, ordered selected predictor names
selection_meta_df = pd.DataFrame([
    {
        'var': r['var'],
        'fold': r['fold'],
        'model': r['model'],
        'params': r['params'],
        'target_logged': r['target_logged'],
        'selected_predictors': r['selected_predictors'],
    }
    for r in results_flat
])
# Full grid: len(vars_order) * n_folds * n_models rows (fewer if a model never converges)
print(f"Selection metadata rows: {len(selection_meta_df)} (max {len(vars_order) * 10 * len(MODEL_NAMES)})")

results_df = pd.DataFrame([
    {**{k: v for k, v in r.items() if k not in ('pred_train', 'pred_val', 'pred_test', 'train_idx', 'val_idx', 'test_idx')},
     'params': str(r['params'])}
    for r in results_flat
])
with open("5_selection_meta.pkl", "wb") as f:
    pickle.dump(selection_meta_df, f)
print(f"Saved selection metadata to 5_selection_meta.pkl ({len(selection_meta_df)} rows)")

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
