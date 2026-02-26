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
from config import MAOC_config, MIC_config, BD_config, pred_config_MAOC_2015, pred_config_MAOC_2018, pred_config_Cmic_2015, pred_config_BD_2015, n_folds_outter
# import custom functions and models
spec_utils = importlib.util.spec_from_file_location("utils_module", "5_utils.py")
utils_module = importlib.util.module_from_spec(spec_utils)
spec_utils.loader.exec_module(utils_module)
find = utils_module.find
apply_models = utils_module.apply_models
haversine = utils_module.haversine
compute_pred_median = utils_module.compute_pred_median

os.makedirs("5_results", exist_ok=True)

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

# ==================================== predict MAOC, MIC & BD per year ====================================
results = find(df, BD_config) # identify n models to predict BD via CV
df = apply_models(df, results, pred_config_BD_2015) # predict BD at 2015 (n predictions per sample)
df = apply_models(df, results, BD_config) # predict BD at 2018 (n predictions per sample)
df = compute_pred_median(df, "BD 0-20_", [2015, 2018])

results = find(df, MAOC_config) # identify n models to predict MAOC via CV
df = apply_models(df, results, pred_config_MAOC_2015) # predict MAOC at 2015 (n predictions per sample)
df = apply_models(df, results, pred_config_MAOC_2018) # predict MAOC at 2018 (n predictions per sample)
df = compute_pred_median(df, "OC_sc_g_kg_", [2015, 2018])

results = find(df, MIC_config) # identify n models to predict MIC via CV
df = apply_models(df, results, pred_config_Cmic_2015) # predict MIC at 2015 (n predictions per sample)
df = apply_models(df, results, MIC_config) # predict MIC at 2018 (n predictions per sample)
df = compute_pred_median(df, "Cmic_", [2015, 2018])

df.to_pickle("5_with_predictions.pkl") # save data with predictions
print(f"Total execution time: {time.time() - start_time:.2f} seconds")