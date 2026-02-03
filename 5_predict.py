# ==================================== import ====================================
import pickle
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
from sklearn.model_selection import train_test_split, KFold
seed = 42
# import custom functions and models
spec_utils = importlib.util.spec_from_file_location("utils_module", "5_utils.py")
utils_module = importlib.util.module_from_spec(spec_utils)
spec_utils.loader.exec_module(utils_module)
find = utils_module.find
apply_models = utils_module.apply_models
haversine = utils_module.haversine
compute_pred_median = utils_module.compute_pred_median
# import configs for model finding and prediction (which features and models to try)
spec_config = importlib.util.spec_from_file_location("config_module", "5_config.py")
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)
models = config_module.models
n_folds_outter = config_module.n_folds_outter
MAOC_config = config_module.MAOC_config
MIC_config = config_module.MIC_config
BD_config = config_module.BD_config
OC_2015_config = config_module.OC_2015_config
OC_2018_config = config_module.OC_2018_config
dOC_config = config_module.dOC_config
dMAOC_config = config_module.dMAOC_config
dMIC_config = config_module.dMIC_config
pred_config_OC_2009 = config_module.pred_config_OC_2009
pred_config_OC_2015 = config_module.pred_config_OC_2015
pred_config_OC_2018 = config_module.pred_config_OC_2018
pred_config_Cmic_2009 = config_module.pred_config_Cmic_2009
pred_config_Cmic_2015 = config_module.pred_config_Cmic_2015
pred_config_Cmic_2018 = config_module.pred_config_Cmic_2018
pred_config_BD_2009 = config_module.pred_config_BD_2009
pred_config_BD_2015 = config_module.pred_config_BD_2015
pred_config_BD_2018 = config_module.pred_config_BD_2018
pred_config_OC_2015_2009 = config_module.pred_config_OC_2015_2009
pred_config_OC_2015_2015 = config_module.pred_config_OC_2015_2015
pred_config_OC_2015_2018 = config_module.pred_config_OC_2015_2018
pred_config_OC_2018_2009 = config_module.pred_config_OC_2018_2009
pred_config_OC_2018_2015 = config_module.pred_config_OC_2018_2015
pred_config_OC_2018_2018 = config_module.pred_config_OC_2018_2018
pred_config_dOC_15_18 = config_module.pred_config_dOC_15_18
pred_config_dOC_sc_g_kg_15_18_median = config_module.pred_config_dOC_sc_g_kg_15_18_median
pred_config_dCmic_15_18_median = config_module.pred_config_dCmic_15_18_median
# load data
with open("4_with_nc.pkl", "rb") as f:
    df = pickle.load(f)
start_time = time.time()

# ==================================== precompute outer splits ====================================
split_col = "split"
all_indices = np.arange(len(df))
train_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=seed, shuffle=True)
split_labels = np.full(len(df), None, dtype=object)
split_labels[test_idx] = 'test'
for fold_id, (_, val_idx) in enumerate(KFold(n_splits=n_folds_outter, shuffle=True, random_state=seed).split(train_idx)):
    split_labels[train_idx[val_idx]] = fold_id
df = pd.concat([df, pd.Series(split_labels, name=split_col)], axis=1)

# ==================================== predict SOC, MAOC, MIC & BD per year ====================================
results = find(df, OC_2015_config) # identify n models to predict OC_2015 via CV
df = apply_models(df, results, pred_config_OC_2015_2009) # predict OC_2015 using 2009 predictors (n predictions per sample)
df = apply_models(df, results, pred_config_OC_2015_2015) # predict OC_2015 using 2015 predictors (n predictions per sample)
df = apply_models(df, results, pred_config_OC_2015_2018) # predict OC_2015 using 2018 predictors (n predictions per sample)
# Compute medians for OC_2015 predictions
for pred_year in [2009, 2015, 2018]:
    col_base = f"OC_2015_{pred_year}_pred_"
    pred_cols = [col for col in df.columns if col.startswith(col_base) and col[len(col_base):].isdigit()]
    pred_cols.sort(key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)
    if pred_cols:
        df[f"OC_2015_{pred_year}_pred_median"] = df[pred_cols].median(axis=1, skipna=True)
results = find(df, OC_2018_config) # identify n models to predict OC_2018 via CV
df = apply_models(df, results, pred_config_OC_2018_2009) # predict OC_2018 using 2009 predictors (n predictions per sample)
df = apply_models(df, results, pred_config_OC_2018_2015) # predict OC_2018 using 2015 predictors (n predictions per sample)
df = apply_models(df, results, pred_config_OC_2018_2018) # predict OC_2018 using 2018 predictors (n predictions per sample)
# Compute medians for OC_2018 predictions
for pred_year in [2009, 2015, 2018]:
    col_base = f"OC_2018_{pred_year}_pred_"
    pred_cols = [col for col in df.columns if col.startswith(col_base) and col[len(col_base):].isdigit()]
    pred_cols.sort(key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)
    if pred_cols:
        df[f"OC_2018_{pred_year}_pred_median"] = df[pred_cols].median(axis=1, skipna=True)
results = find(df, BD_config) # identify n models to predict BD via CV
df = apply_models(df, results, pred_config_BD_2009) # predict BD at 2009 (n predictions per sample)
df = apply_models(df, results, pred_config_BD_2015) # predict BD at 2015 (n predictions per sample)
df = apply_models(df, results, pred_config_BD_2018) # predict BD at 2018 (n predictions per sample)
compute_pred_median(df, "BD 0-20_", [2009, 2015, 2018])
results = find(df, MAOC_config) # identify n models to predict MAOC via CV
df = apply_models(df, results, pred_config_OC_2009) # predict MAOC at 2009 (n predictions per sample)
df = apply_models(df, results, pred_config_OC_2015) # predict MAOC at 2015 (n predictions per sample)
df = apply_models(df, results, pred_config_OC_2018) # predict MAOC at 2018 (n predictions per sample)
compute_pred_median(df, "OC_sc_g_kg_", [2009, 2015, 2018])
results = find(df, MIC_config) # identify n models to predict MIC via CV
df = apply_models(df, results, pred_config_Cmic_2009) # predict MIC at 2009 (n predictions per sample)
df = apply_models(df, results, pred_config_Cmic_2015) # predict MIC at 2015 (n predictions per sample)
df = apply_models(df, results, pred_config_Cmic_2018) # predict MIC at 2018 (n predictions per sample)
compute_pred_median(df, "Cmic_", [2009, 2015, 2018])

# ==================================== calculate changes of SOC, MAOC & MIC ====================================
df['gps_dist_2015_2018_m'] = haversine(df['gps_lat_2015'], df['gps_long_2015'],
    df['gps_lat_2018'], df['gps_long_2018']) # calculate distance between 2015 and 2018 LUCAS locations
print("Distances calculated:", df['gps_dist_2015_2018_m'].notna().sum())
df.loc[df['gps_dist_2015_2018_m'] > 10, 'gps_dist_2015_2018_m'] = np.nan # filter out distances > 10 m
print("Distances after filtering:", df['gps_dist_2015_2018_m'].notna().sum())
# plot distribution of remaining distances
os.makedirs("5_results", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.hist(df['gps_dist_2015_2018_m'].dropna(), bins=40, alpha=0.7, color='skyblue')
plt.xlabel('Distance between 2015 and 2018 GPS (meters)')
plt.ylabel('Frequency')
plt.title('Distribution of GPS Distance (2015-2018)')
plt.tight_layout()
plt.savefig("5_results/gps_distance_distribution_2015_2018.png")
plt.close()
# calculate change of SOC, MAOC and MIC between 2015 and 2018 (n*n changes)
mask_dist_exists = df['gps_dist_2015_2018_m'].notna() # create a mask where the distance exists
df = df.drop(columns=['dOC_15_18'], errors='ignore')
df['dOC_15_18'] = np.where(mask_dist_exists, df['OC_2018'] - df['OC_2015'], np.nan)
new_columns = {}
for i in range(n_folds_outter):
    for j in range(n_folds_outter):
        new_columns[f'dOC_sc_g_kg_15_18_{i}_{j}'] = np.where(mask_dist_exists, df[f'OC_sc_g_kg_2018_pred_{i}'] - df[f'OC_sc_g_kg_2015_pred_{j}'], np.nan)
        new_columns[f'dCmic_15_18_{i}_{j}'] = np.where(mask_dist_exists, df[f'Cmic_2018_pred_{i}'] - df[f'Cmic_2015_pred_{j}'], np.nan)
df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
# Calculate the median across all n*n changes for each sample
dOC_sc_g_kg_15_18_cols = [f'dOC_sc_g_kg_15_18_{i}_{j}' for i in range(n_folds_outter) for j in range(n_folds_outter)]
dCmic_15_18_cols = [f'dCmic_15_18_{i}_{j}' for i in range(n_folds_outter) for j in range(n_folds_outter)]
df['dOC_sc_g_kg_15_18_median'] = df[dOC_sc_g_kg_15_18_cols].median(axis=1, skipna=True)
df['dCmic_15_18_median'] = df[dCmic_15_18_cols].median(axis=1, skipna=True)

# ==================================== predict changes of SOC, MAOC & MIC ====================================
results = find(df, dOC_config)
df = apply_models(df, results, pred_config_dOC_15_18)
results = find(df, dMAOC_config)
df = apply_models(df, results, pred_config_dOC_sc_g_kg_15_18_median)
results = find(df, dMIC_config)
df = apply_models(df, results, pred_config_dCmic_15_18_median)

df.to_pickle("5_with_predictions.pkl") # save data with predictions
print(f"Total execution time: {time.time() - start_time:.2f} seconds")