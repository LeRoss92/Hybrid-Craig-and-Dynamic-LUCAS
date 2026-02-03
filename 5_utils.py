import numpy as np
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 warnings
import matplotlib.pyplot as plt
# Suppress numpy division warnings in correlation calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
# Also suppress warnings from numpy's corrcoef function
np.seterr(divide='ignore', invalid='ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.base import clone
from joblib import Parallel, delayed
import importlib.util

spec = importlib.util.spec_from_file_location("models_module", "5_models.py")
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
get_models = models_module.get_models

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

def process_fold(fold_idx, X_train_all, y_train_all, split_train_labels, X_test, y_test, target_name, feature_names, config, all_MODELS, fold_values):
    """Process a single fold - can be called in parallel"""
    # Set matplotlib backend in worker process to avoid X11 warnings
    import matplotlib
    matplotlib.use('Agg', force=True)
    
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
    print(f"TARGET: {target_name} | Train={len(X_train_all)} | Test={len(X_test)} | Fold {fold_idx}")
    
    fold_results = []
    for model_type in config['models']:
            # -------------------- Additive Feature Selection --------------------
            selected_idx = [] # store selected (best in each round)
            remaining_idx = list(range(X_train_p.shape[1])) # initialize possible features to select
            model_config = all_MODELS[model_type] # get model info
            kge_scorer = make_scorer(kge, greater_is_better=True) # use KGE for selection
            best_overall_score, best_overall_model, best_overall_features, best_overall_hyperparams = -np.inf, None, None, None # initialize to get the best model at best feature selection
            n_jobs_gridsearch = model_config.get('n_jobs', 1)  # Use n_jobs for GridSearchCV
            
            for _ in range(min(config['max_features'], len(remaining_idx))): # loop until enough features found
                best_idx, best_score = None, -np.inf # initialize to store presently best in this round
                best_search = None # initilize to store best of this features selection round
                for cand_idx in remaining_idx: # loop over possibles to select
                    feat_set = selected_idx + [cand_idx] # add respective possible to select
                    search = GridSearchCV(model_config['model'], model_config['params'], cv=config['n_folds_HP_opt'], scoring=kge_scorer, n_jobs=n_jobs_gridsearch, verbose=0) # set up search
                    search.fit(X_train_p[:, feat_set], y_train_p) # find best model
                    if search.best_score_ > best_score: # if new best (of this feature selection round)
                        best_score, best_idx, best_search = search.best_score_, cand_idx, search # than replace with this
                
                selected_idx.append(best_idx) # select
                remaining_idx.remove(best_idx) # remove selected from the futures possible to select
                if best_score > best_overall_score: # if new best (of this fold + model)
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

def find(df, config, split_col="split"):
    results = pd.DataFrame()
    target_name = config["target_name"]
    all_MODELS = get_models(config['seed'])
    feature_names = config['predictors']["normal"] + config['predictors']["log"]
    X_raw, y, valid_mask = prepare_data(df, target_name, feature_names) # filter for valid samples (with y)
    X_raw[:, len(config['predictors']["normal"]):] = np.log1p(X_raw[:, len(config['predictors']["normal"]):])
    split_labels = np.array(df.loc[valid_mask, split_col]) # get the split labels (only where target exists)
    test_mask = split_labels == 'test' # identify final test samples
    X_test, y_test = X_raw[test_mask], y[test_mask] # get test data
    X_train_all, y_train_all = X_raw[~test_mask], y[~test_mask] # get train/eval samples
    split_train_labels = split_labels[~test_mask] # remove test
    fold_values = sorted({int(v) for v in pd.unique(split_train_labels) if v != 'test'}) # sort indices according to fold

    # ==================================== Outer CV using precomputed folds - PARALLELIZED ====================================
    n_jobs_folds = config.get('n_jobs_folds', 1)  # Number of folds to process in parallel
    if n_jobs_folds > 1:
        # Parallelize fold processing
        fold_results_list = Parallel(n_jobs=n_jobs_folds, backend='loky')(
            delayed(process_fold)(fold_idx, X_train_all, y_train_all, split_train_labels, X_test, y_test, target_name, feature_names, config, all_MODELS, fold_values)
            for fold_idx in fold_values
        )
        # Flatten results from all folds
        for fold_results in fold_results_list:
            results = pd.concat([results, pd.DataFrame(fold_results)], ignore_index=True)
    else:
        # Sequential processing
        for fold_idx in fold_values:
            fold_results = process_fold(fold_idx, X_train_all, y_train_all, split_train_labels, X_test, y_test, target_name, feature_names, config, all_MODELS, fold_values)
            results = pd.concat([results, pd.DataFrame(fold_results)], ignore_index=True)
    
    results.to_pickle(f'5_results/5_{target_name}_results.pkl')
    return results

def apply_models(df, results, pred_config):
    pred_columns = {} # to store predictions
    for fold_idx in results['fold'].unique(): # one prediction per fold (uncertainty)
        fold_results = results[results['fold'] == fold_idx] # filter to have evaluation results only of that fold
        best_model_row = fold_results.loc[fold_results['val_KGE'].idxmax()] # identify best model (via val, not test!)
        model = best_model_row['model'] # get model, preprocessors and indices of the features from selection
        preprocessor = best_model_row['preprocessor']
        target_scaler = best_model_row['target_scaler']
        feature_indices = best_model_row['feature_indices']
        feature_names = pred_config['predictors']['normal'] + pred_config['predictors']['log'] # get new feature names (of the year to predict)
        X_raw = np.column_stack([get_array(df, col) for col in feature_names]) # with the indices of the old features (where model was trained on) identify new features to use
        X_raw[:, len(pred_config['predictors']['normal']):] = np.log1p(X_raw[:, len(pred_config['predictors']['normal']):]) # log the features to log
        X_p = preprocessor.transform(X_raw) # impute and zscore (with statistics of model training)
        y_pred_scaled = model.predict(X_p[:, feature_indices]) # predict
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten() # re-scale predictions to original scale
        if pred_config['target_log']: # re-log if target was logged
            y_pred = np.expm1(y_pred)
        pred_columns[f"{pred_config['target_name']}_pred_{fold_idx}"] = y_pred # store predictions
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

def compute_pred_median(df, prefix, years):
    """
    For each year, find prediction columns by prefix + year, compute median, add as new column.
    E.g., prefix="OC_sc_g_kg_", years=[2009,2015,2018] looks for "OC_sc_g_kg_2009_pred_*" columns.
    """
    for year in years:
        col_base = f"{prefix}{year}_pred_"
        pred_cols = [col for col in df.columns if col.startswith(col_base) and col[len(col_base):].isdigit()]
        # Sort by integer suffix, if present
        pred_cols.sort(key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)
        median_col = f"{prefix}{year}_pred_median"
        df[median_col] = df[pred_cols].median(axis=1, skipna=True)