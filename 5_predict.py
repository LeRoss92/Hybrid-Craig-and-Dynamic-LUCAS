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

def resolve_inference_spec(parent_config, inf_spec):
    """Merge a TARGET_CONFIG['inference'] entry with the parent; omitted keys inherit from parent."""
    return {
        "target_name": inf_spec["target_name"],
        "predictors": tuple(inf_spec.get("predictors", parent_config["predictors"])),
        "log_predictors": tuple(inf_spec.get("log_predictors", parent_config["log_predictors"])),
        "categoricals": list(inf_spec.get("categoricals", parent_config.get("categoricals", []))),
    }

def _r2_masked(y, pred):
    if y is None:
        return np.nan
    m = np.isfinite(np.asarray(y)) & np.isfinite(np.asarray(pred))
    if m.sum() < 2:
        return np.nan
    return r2_score(np.asarray(y)[m], np.asarray(pred)[m])

def _safe_infer_col_suffix(target_name):
    """Column-safe fragment for inference target name in pred_* column names."""
    return str(target_name).replace(" ", "_").replace("/", "_")

def process_one_fold(var, fold, X_train, y_train, X_val, y_val, X_test, y_test,
                    train_idx, val_idx, test_idx, n_normal, n_log, pred_cols, inference_bundles,
                    X_train_all, X_val_all, X_test_all, y_train_all):
    """Process a single (var, fold): fit on rows with y; metrics on those rows; predictions on all rows per split."""
    def to_signed_log(y): # log the abs value and give it sign: log of negative (changes) makes problems otherwise
        return np.sign(y) * np.log1p(np.abs(y))
    def from_signed_log(y_log): # function to revert (when prediction)
        y_log = np.clip(y_log, -np.log(np.finfo(np.float64).max), np.log(np.finfo(np.float64).max))
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
    for model_name in TRAIN_DEFAULTS['models']: # loop over methods (LR, PLR, XGB...)
        model_config = models[model_name] # get info on model
        param_grid = model_config.get('params', {}) or {} # create latin hypercube of hyperparameter combinations (if HP opt...)
        best_val_r2 = -np.inf # init best metric (worst R2 in -inf)
        best_result = None
        for params_combo in ParameterGrid(param_grid) if param_grid else [{}]: # loop over hyperparameter combos
            try:
                # Option 1: raw target
                selected_raw = forward_select(X_train_p, X_val_p, y_train, y_val, model_config, params_combo, TRAIN_DEFAULTS['max_features']) # select predictors with normal target: indices of selected returned
                m_raw = _make_model(model_config, params_combo, n_features=len(selected_raw)) # get the model (seems to make sense)
                m_raw.fit(X_train_p[:, selected_raw], y_train) # fit the model to selected predictors (on train only)
                pred_train_raw = m_raw.predict(X_train_p[:, selected_raw]) # predict train
                pred_val_raw = m_raw.predict(X_val_p[:, selected_raw]) # predict val
                pred_test_raw = m_raw.predict(X_test_p[:, selected_raw]) # predict test
                val_r2_raw = r2_score(y_val, pred_val_raw) # calculate val R2
                train_r2_raw = r2_score(y_train, pred_train_raw) # calculate train R2
                test_r2_raw = r2_score(y_test, pred_test_raw) # calculate test R2
                # Option 2: log target -> do same with logged target
                selected_log = forward_select(X_train_p, X_val_p, y_train_log, y_val_log, model_config, params_combo, TRAIN_DEFAULTS['max_features']) # select predictors with logged target: indices of selected returned
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
            tl = best_result['target_logged']
            m_final = _make_model(model_config, best_result['params'], n_features=len(sel))
            if tl:
                m_final.fit(X_train_p[:, sel], to_signed_log(y_train))
            else:
                m_final.fit(X_train_p[:, sel], y_train)

            def _predict_from_Xp(Xp):
                raw = m_final.predict(Xp[:, sel])
                return from_signed_log(raw) if tl else raw

            inference_metrics = []
            inference_preds = []
            for ib in inference_bundles:
                rec = {
                    'inference_target': ib['target_name'],
                    'train r2': np.nan,
                    'val r2': np.nan,
                    'test r2': np.nan,
                    'frozen_main_model': False,
                }
                pred_pack = {
                    'inference_target': ib['target_name'],
                    'pred_train': None,
                    'pred_val': None,
                    'pred_test': None,
                }
                try:
                    Xtr = np.asarray(ib['X_train'], dtype=float, order='C').copy()
                    Xva = np.asarray(ib['X_val'], dtype=float, order='C').copy()
                    Xte = np.asarray(ib['X_test'], dtype=float, order='C').copy()
                    nn, nl = ib['n_normal'], ib['n_log']
                    Xtr[:, nn:nn + nl] = np.log1p(Xtr[:, nn:nn + nl])
                    Xva[:, nn:nn + nl] = np.log1p(Xva[:, nn:nn + nl])
                    Xte[:, nn:nn + nl] = np.log1p(Xte[:, nn:nn + nl])
                    ytr_i, yva_i, yte_i = ib['y_train'], ib['y_val'], ib['y_test']
                    if Xtr.shape[1] == X_train.shape[1]:
                        Xtr_p = preprocessor.transform(Xtr)
                        Xva_p = preprocessor.transform(Xva)
                        Xte_p = preprocessor.transform(Xte)
                        rec['frozen_main_model'] = True
                        ptr = _predict_from_Xp(Xtr_p)
                        pva = _predict_from_Xp(Xva_p)
                        pte = _predict_from_Xp(Xte_p)
                    else:
                        prep_i = Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ])
                        Xtr_p = prep_i.fit_transform(Xtr)
                        Xva_p = prep_i.transform(Xva)
                        Xte_p = prep_i.transform(Xte)
                        m_i = _make_model(model_config, best_result['params'], n_features=len(sel))
                        # Refit on inference-year X; use inference y only if present (≥2 finite), else main y_train
                        use_inf_y = ytr_i is not None and np.isfinite(ytr_i).sum() >= 2
                        y_fit_source = ytr_i if use_inf_y else y_train_all
                        y_fit = to_signed_log(y_fit_source) if tl else y_fit_source
                        fit_m = np.isfinite(y_fit) & np.all(np.isfinite(Xtr_p[:, sel]), axis=1)
                        if fit_m.sum() < 2:
                            inference_metrics.append(rec)
                            inference_preds.append(pred_pack)
                            continue
                        m_i.fit(Xtr_p[fit_m][:, sel], y_fit[fit_m])
                        def _pred_i(Xp):
                            r = m_i.predict(Xp[:, sel])
                            return from_signed_log(r) if tl else r
                        ptr, pva, pte = _pred_i(Xtr_p), _pred_i(Xva_p), _pred_i(Xte_p)
                    rec['train r2'] = _r2_masked(ytr_i, ptr)
                    rec['val r2'] = _r2_masked(yva_i, pva)
                    rec['test r2'] = _r2_masked(yte_i, pte)
                    pred_pack['pred_train'] = ptr
                    pred_pack['pred_val'] = pva
                    pred_pack['pred_test'] = pte
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    pass
                inference_metrics.append(rec)
                inference_preds.append(pred_pack)

            def _prep_full_X(X_raw):
                X = np.asarray(X_raw, dtype=float, order='C').copy()
                X[:, n_normal:n_normal + n_log] = np.log1p(X[:, n_normal:n_normal + n_log])
                return preprocessor.transform(X)

            pred_train = _predict_from_Xp(_prep_full_X(X_train_all))
            pred_val = _predict_from_Xp(_prep_full_X(X_val_all))
            pred_test = _predict_from_Xp(_prep_full_X(X_test_all))

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
                'pred_train': pred_train,
                'pred_val': pred_val,
                'pred_test': pred_test,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                'selected': sel,
                'inference_metrics': inference_metrics,
                'inference_preds': inference_preds,
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

def _y_column_if_present(frame, col_name):
    """Return float numpy y for rows in frame, or None if column missing (no ground truth for that target)."""
    if col_name not in frame.columns:
        return None
    return frame[col_name].to_numpy(dtype=float)

######################################## create parallel tasks ########################################
tasks = []
for var in TARGET_CONFIG.keys():
    config = TARGET_CONFIG[var] # info (targets, predictors, inference, log...)
    target_name = config["target_name"] # target name
    onehot_cols, _ = get_onehot_cols_for_config(df, config) # get data of 1-hot variables
    pred_cols = list(config['predictors']) + list(config['log_predictors']) + onehot_cols # all potential predictors together (1-hot might get selected individually)
    n_normal = len(config['predictors']) # number of normal distributed predictors
    n_log = len(config['log_predictors']) # numer of log normal distributed predictors
    test_data = df[(df['split'] == 'test') & (df[target_name].notna())] # valid test rows
    X_test, y_test  = test_data[pred_cols].to_numpy(dtype=float), test_data[target_name].to_numpy(dtype=float) # get test X and y as array
    X_test[:, n_normal:n_normal + n_log] = np.log1p(X_test[:, n_normal:n_normal + n_log]) # log respective predictors (looks correct) 
    for fold in range(10): # loop over folds
        train_fit = df[(df['split'] != 'test') & (df['split'] != fold) & (df[target_name].notna())]
        val_fit = df[(df['split'] != 'test') & (df['split'] == fold) & (df[target_name].notna())]
        train_all = df[(df['split'] != 'test') & (df['split'] != fold)]
        val_all = df[(df['split'] != 'test') & (df['split'] == fold)]
        test_all = df[(df['split'] == 'test')]
        train_idx = train_all.index.to_numpy()
        val_idx = val_all.index.to_numpy()
        test_idx = test_all.index.to_numpy()
        X_train = train_fit[pred_cols].to_numpy(dtype=float)
        y_train = train_fit[target_name].to_numpy(dtype=float)
        X_val = val_fit[pred_cols].to_numpy(dtype=float)
        y_val = val_fit[target_name].to_numpy(dtype=float)
        X_train_all = train_all[pred_cols].to_numpy(dtype=float)
        X_val_all = val_all[pred_cols].to_numpy(dtype=float)
        X_test_all = test_all[pred_cols].to_numpy(dtype=float)
        y_train_all = train_all[target_name].to_numpy(dtype=float)
        inference_bundles = []
        for raw_inf in config.get('inference') or []:
            inf = resolve_inference_spec(config, raw_inf)
            oh_i, _ = get_onehot_cols_for_config(df, inf)
            pc_i = list(inf['predictors']) + list(inf['log_predictors']) + oh_i
            nn_i = len(inf['predictors'])
            nl_i = len(inf['log_predictors'])
            tn_i = inf['target_name']
            inference_bundles.append({
                'target_name': tn_i,
                'X_train': train_all[pc_i].to_numpy(dtype=float),
                'X_val': val_all[pc_i].to_numpy(dtype=float),
                'X_test': test_all[pc_i].to_numpy(dtype=float),
                'y_train': _y_column_if_present(train_all, tn_i),
                'y_val': _y_column_if_present(val_all, tn_i),
                'y_test': _y_column_if_present(test_all, tn_i),
                'n_normal': nn_i,
                'n_log': nl_i,
            })
        tasks.append((var, fold, X_train, y_train, X_val, y_val, X_test.copy(), y_test.copy(),
                      train_idx, val_idx, test_idx, n_normal, n_log, pred_cols, inference_bundles,
                      X_train_all, X_val_all, X_test_all, y_train_all))

######################################## compute ########################################
results = Parallel(n_jobs=TRAIN_DEFAULTS['N_JOBS'], backend='loky')(delayed(process_one_fold)(*task) for task in tasks)

######################################## post processing ########################################
# Each fold returns a list of dicts (one per model); flatten
results_flat = [row for fold_results in results for row in fold_results]

# One row per (var, fold, model): hyperparameters dict, log-target flag, ordered selected predictor names
selection_meta_df = pd.DataFrame([
    {
        'var': r['var'],
        'fold': r['fold'],
        'model': r['model'],
        'params': r['params'],
        'target_logged': r['target_logged'],
        'selected_predictors': r['selected_predictors'],
        'inference_metrics': r.get('inference_metrics', []),
    }
    for r in results_flat
])

_results_exclude = (
    'pred_train', 'pred_val', 'pred_test', 'train_idx', 'val_idx', 'test_idx', 'inference_preds',
)
results_df = pd.DataFrame([
    {**{k: v for k, v in r.items() if k not in _results_exclude},
     'params': str(r['params'])}
    for r in results_flat
])
# Long-form inference rows (one row per var/fold/model/inference_target) for filtering and tables
inf_rows = []
for r in results_flat:
    for im in r.get('inference_metrics') or []:
        inf_rows.append({
            'var': r['var'],
            'fold': r['fold'],
            'model': r['model'],
            'inference_target': im['inference_target'],
            'inference_train_r2': im['train r2'],
            'inference_val_r2': im['val r2'],
            'inference_test_r2': im['test r2'],
            'inference_frozen_main_model': im.get('frozen_main_model', False),
        })
inference_results_df = pd.DataFrame(inf_rows)
with open("5_selection_meta.pkl", "wb") as f:
    pickle.dump(selection_meta_df, f)
with open("5_results_df.pkl", "wb") as f:
    pickle.dump({"results_df": results_df, "inference_by_target": inference_results_df}, f)

# Build output df: original df + one column per (var, fold, model) with train/val/test predictions
# on every row in each split (not only rows with observed target). Inference preds use the same splits.
out_df = df.copy()
pred_cols = []
for r in results_flat:
    pred_cols.append(f"pred_{r['var']}_fold{r['fold']}_{r['model']}")
    pred_cols.extend(
        f"pred_{r['var']}_fold{r['fold']}_{r['model']}_inf_{_safe_infer_col_suffix(ip['inference_target'])}"
        for ip in (r.get('inference_preds') or [])
        if ip.get('pred_train') is not None
    )
out_df = pd.concat([out_df, pd.DataFrame(np.nan, index=out_df.index, columns=pred_cols)], axis=1)
for r in results_flat:
    col = f"pred_{r['var']}_fold{r['fold']}_{r['model']}"
    out_df.loc[r['train_idx'], col] = r['pred_train']
    out_df.loc[r['val_idx'], col] = r['pred_val']
    out_df.loc[r['test_idx'], col] = r['pred_test']
    for ip in r.get('inference_preds') or []:
        if ip.get('pred_train') is None:
            continue
        suf = _safe_infer_col_suffix(ip['inference_target'])
        icol = f"pred_{r['var']}_fold{r['fold']}_{r['model']}_inf_{suf}"
        out_df.loc[r['train_idx'], icol] = ip['pred_train']
        out_df.loc[r['val_idx'], icol] = ip['pred_val']
        out_df.loc[r['test_idx'], icol] = ip['pred_test']
with open("5_with_predictions.pkl", "wb") as f:
    pickle.dump(out_df, f)
_pred_cols = [c for c in out_df.columns if c.startswith('pred_')]
_n_inf = sum(1 for c in _pred_cols if '_inf_' in c)

import matplotlib.pyplot as plt
model_order = [m for m in TRAIN_DEFAULTS['models'] if m in results_df['model'].values]
combo_order = [f"{v}-{m}" for v in TARGET_CONFIG.keys() for m in model_order]
data = [
    results_df[(results_df['var'] == v) & (results_df['model'] == m)]['test r2'].values
    for v in TARGET_CONFIG.keys() for m in model_order
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
