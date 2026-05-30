# Predictor groups
pred_groups = {
    'Texture': {
        'color': '#f5f5dc',  # Beige
        'average': {
            'Clay': 'Clay',
            'Clay+Silt': 'ClaySilt', 
            'Coarse': 'Coarse', 
            'Stones': 'soil_stones_perc_avg_09_15_18',
        },
        '2009': {
            'Stones': 'soil_stones_perc_2009',
        },
        '2015': {
            'Stones': 'soil_stones_perc_2015',
        },
        '2018': {
            'Stones': 'soil_stones_perc_2018',
        },
        'change': {
            'Stones': 'soil_stones_perc_linreg_slope',
        },
    },
    'Bulk Density': {
        'color': '#f5f5dc',  # Beige
        '2018': {
            'Bulk Density': 'BD 0-20_2018',
        },
    },
    'Mineralogy': {
        'color': 'grey',
        'average': {
            'Al': 'Ox_Al_2018', 
            'Fe': 'Ox_Fe_2018'
        },
    },
    'Nutrients': {
        'color': 'purple',
        '2009': {
            'CN': 'CN_2009', 
            'CP': 'CP_2009', 
            'CK': 'CK_2009', 
        },
        '2015': {
            'CN': 'CN_2015', 
            'CP': 'CP_2015', 
            'CK': 'CK_2015', 
        },
        '2018': {
            'CN': 'CN_2018', 
            'CP': 'CP_2018', 
            'CK': 'CK_2018', 
        },
        'average': {
            'CN': 'CN_avg_09_15_18', 
            'CP': 'CP_avg_09_15_18', 
            'CK': 'CK_avg_09_15_18', 
        },
        'change': {
            'CN': 'CN_linreg_slope', 
            'CP': 'CP_linreg_slope', 
            'CK': 'CK_linreg_slope', 
        },
    },
    'Acidity': {
        'color': 'orange',
        '2009': {
            'pH CaCl2': 'pH_CaCl2_2009',
            'pH H2O': 'pH_H2O_2009',
            'CaCO3': 'CaCO3_2009',
        },
        '2015': {
            'pH CaCl2': 'pH_CaCl2_2015',
            'pH H2O': 'pH_H2O_2015',
            'CaCO3': 'CaCO3_2015',
        },
        '2018': {
            'pH CaCl2': 'pH_CaCl2_2018',
            'pH H2O': 'pH_H2O_2018',
            'CaCO3': 'CaCO3_2018',
        },
        'average': {
            'pH CaCl2': 'pH_CaCl2_avg_09_15_18',
            'pH H2O': 'pH_H2O_avg_09_15_18',
            'CaCO3': 'CaCO3_avg_09_15_18',
        },
        'change': {
            'pH CaCl2': 'pH_CaCl2_linreg_slope',
            'pH H2O': 'pH_H2O_linreg_slope',
            'CaCO3': 'CaCO3_linreg_slope',
        },
    },
    'SOC': {
        'color': '#deb887',  # Light brown
        '2009': {
            'SOC': 'OC_2009',
        },
        '2015': {
            'SOC': 'OC_2015',
        },
        '2018': {
            'SOC': 'OC_2018',
        },
        'average': {
            'SOC': 'OC_avg_09_15_18',
        },
        'change': {
            'SOC': 'OC_linreg_slope',
        },
    },
    'Season': {
        'color': '#add8e6',  # light blue
        '2009': {
            'DOY': 'doy_2009',
        },
        '2015': {
            'DOY': 'doy_2015',
        },
        '2018': {
            'DOY': 'doy_2018',
        },
        'average': {
            'DOY': 'doy_avg_09_15_18',
        },
        'change': {
            'DOY': 'doy_linreg_slope',
        },
    },
    'LC': {
        'color': '#c6efce',  # Light green
        '2009': {
            'Grassland': 'lc1_2_2009_E',
            'Cropland': 'lc1_2_2009_B',
            'Woodland': 'lc1_2_2009_C',
            'Grazing': 'grazing_2009',
            'Irrigation': 'wm_2009',
            # 'Residues': 'soil_crop_2009', # too sparse
            'tree height': 'tree_height_survey_2009'
        },
        '2015': {
            'Grassland': 'lc1_2_2015_E',
            'Cropland': 'lc1_2_2015_B',
            'Woodland': 'lc1_2_2015_C',
            'Grazing': 'grazing_2015',
            'Irrigation': 'wm_2015',
            # 'Residues': 'soil_crop_2015',
            'tree height': 'tree_height_survey_2015'
        },
        '2018': {
            'Grassland': 'lc1_2_2015_E',
            'Cropland': 'lc1_2_2015_B',
            'Woodland': 'lc1_2_2015_C',
            'Grazing': 'grazing_2018',
            'Irrigation': 'wm_2018',
            # 'Residues': 'soil_crop_2018',
            'tree height': 'tree_height_survey_2018'
        },
        'average': {
            'Grassland': 'E_avg_09_15_18',
            'Cropland': 'B_avg_09_15_18',
            'Woodland': 'C_avg_09_15_18',
            'Grazing': 'grazing_avg_09_15_18',
            'Irrigation': 'wm_avg_09_15_18',
            'Residues': 'soil_crop_avg_09_15_18',
            'tree height': 'tree_height_survey_avg_09_15_18'
        },
        'change': {
            'Grassland': 'E_linreg_slope',
            'Cropland': 'B_linreg_slope',
            'Woodland': 'C_linreg_slope',
            'Grazing': 'grazing_linreg_slope',
            'Irrigation': 'wm_linreg_slope',
            'Residues': 'soil_crop_linreg_slope',
            'tree height': 'tree_height_survey_linreg_slope'
        },
    },
    'Atmosphere': {
        'color': '#ff9999',  # Light red
        '2009': {
            'H': 'Fluxcom_H_2009-5_mean',
            'LE': 'Fluxcom_LE_2009-5_mean',
            'T': 'era5_land_t2m_2009-5_mean',
            'P': 'era5_land_tp_2009-5_mean',
            'PET': 'era5_land_hpet_2009-5_mean',
            'AI': 'aridity_index_2009-5_mean',
        },
        '2015': {
            'H': 'Fluxcom_H_2015-5_mean',
            'LE': 'Fluxcom_LE_2015-5_mean',
            'T': 'era5_land_t2m_2015-5_mean',
            'P': 'era5_land_tp_2015-5_mean',
            'PET': 'era5_land_hpet_2015-5_mean',
            'AI': 'aridity_index_2015-5_mean',
        },
        '2018': {
            'H': 'Fluxcom_H_2018-5_mean',
            'LE': 'Fluxcom_LE_2018-5_mean',
            'T': 'era5_land_t2m_2018-5_mean',
            'P': 'era5_land_tp_2018-5_mean',
            'PET': 'era5_land_hpet_2018-5_mean',
            'AI': 'aridity_index_2018-5_mean',
        },
        'average': {
            'H': 'Fluxcom_H_avg_09_15_18',
            'LE': 'Fluxcom_LE_avg_09_15_18',
            'T': 'era5_land_t2m_avg_09_15_18',
            'P': 'era5_land_tp_avg_09_15_18',
            'PET': 'era5_land_hpet_avg_09_15_18',
            'AI': 'aridity_index_avg_09_15_18',
        },
        'change': {
            'H': 'Fluxcom_H_linreg_slope',
            'LE': 'Fluxcom_LE_linreg_slope',
            'T': 'era5_land_t2m_linreg_slope',
            'P': 'era5_land_tp_linreg_slope',
            'PET': 'era5_land_hpet_linreg_slope',
            'AI': 'aridity_index_linreg_slope',
        },
    },
    'NPP': {
        'color': 'brown',
        '2009': {
            'NPP': 'MODIS_NPP_2009-5_mean',
        },
        '2015': {
            'NPP': 'MODIS_NPP_2015-5_mean',
        },
        '2018': {
            'NPP': 'MODIS_NPP_2018-5_mean',
        },
        'average': {
            'NPP': 'MODIS_NPP_avg_09_15_18',
        },
        'change': {
            'NPP': 'MODIS_NPP_linreg_slope',
        },
    },
}
to_log = ['CN', 'Clay', 'Coarse', 'Stones', 'Al', 'Fe', 'SOC'] 
to_2log = ['CP', 'CK', 'CaCO3']

# Single dict: targets at first level. Per target: predictors, log_predictors, categoricals, inference.
# inference = list of {target_name, predictors, log_predictors, categoricals} - each item is a full pred_config
TARGET_CONFIG = {
    "MAOCi": {
        "target_name": "MAOC_index_2009",
        "predictor_groups": [
            ("Texture", "average"),
            ("Texture", "2009"),
            ("Bulk Density", "2018"),
            ("Mineralogy", "average"),
            ("Nutrients", "2009"),
            ("Acidity", "2009"),
            ("SOC", "2009"),
            ("Season", "2009"),
            ("LC", "2009"),
            ("Atmosphere", "2009"),
            ("NPP", "2009"),
            ],
        "selected_predictors": ["CN_2009", "lc1_2_2009_B", "OC_2009", "ClaySilt", "era5_land_hpet_2009-5_mean", "Ox_Al_2018"]
    },
    "MICi": {
        "target_name": "Cmic_index_2018",
        "predictor_groups": [
            ("Texture", "average"),
            ("Texture", "2018"),
            ("Bulk Density", "2018"),
            ("Mineralogy", "average"),
            ("Nutrients", "2018"),
            ("Acidity", "2018"),
            ("SOC", "2018"),
            ("Season", "2018"),
            ("LC", "2018"),
            ("Atmosphere", "2018"),
            ("NPP", "2018"),
            ],
        "selected_predictors": ["CN_2018", "pH_CaCl2_2018", "era5_land_hpet_2018-5_mean", "ClaySilt", "OC_2018"]
    },
    "dSOC": {
        "target_name": "OC_linreg_slope",
        "predictor_groups": [
            ("Texture", "average"),
            ("Texture", "change"),
            ("Bulk Density", "2018"),
            ("Mineralogy", "average"),
            ("Nutrients", "average"),
            ("Nutrients", "change"),
            ("Acidity", "average"),
            ("Acidity", "change"),
            ("SOC", "average"),
            ("Season", "average"),
            ("Season", "change"),
            ("LC", "average"),
            ("LC", "change"),
            ("Atmosphere", "average"),
            ("Atmosphere", "change"),
            ("NPP", "average"),
            ("NPP", "change"),
            ],
        "selected_predictors": ["CN_linreg_slope", "Clay", "ClaySilt", "CP_linreg_slope", "OC_avg_09_15_18", "pH_H2O_avg_09_15_18", "pH_CaCl2_avg_09_15_18", "era5_land_hpet_avg_09_15_18"]
    },
    "SOC": {
        "target_name": "OC_avg_09_15_18",
        "predictor_groups": [
            ("Texture", "average"),
            ("Bulk Density", "2018"),
            ("Mineralogy", "average"),
            ("Nutrients", "average"),
            ("Acidity", "average"),
            ("Season", "average"),
            ("LC", "average"),
            ("Atmosphere", "average"),
            ("NPP", "average"),
            ],
        "selected_predictors": ["CP_avg_09_15_18", "CN_avg_09_15_18", "aridity_index_avg_09_15_18", "Clay", "ClaySilt", "pH_H2O_avg_09_15_18", "E_avg_09_15_18"]
    }
}

use_model = {
    'MAOCi': 'XGB-n',
    'MICi': 'XGB-n',
    'dSOC': 'XGB-n',
    'SOC': 'XGB-n',
    # 'SOC09': 'XGB',
    # 'SOC15': 'XGB',
    # 'SOC18': 'XGB'
}

TRAIN_DEFAULTS = {
    'models': ['LinReg', 'XGB-1', 'XGB-n'], # , 'Piecewise_Linear_Reg'
    'seed': 42,
    'max_features': 20,
    'n_folds_HP_opt': 3, 
    'n_jobs_folds': 8,
    'N_JOBS': 70
}

# # Legacy exports for 6_hybrid, 7_analysis, sensitivity_analysis
# predictors_dynamic = (
#     pred_groups['Texture']
#     + pred_groups['Mineral Activity'][2015]
#     + pred_groups['Ox. ex. Al/Fe']
#     + pred_groups['LUCAS normal avg']
#     + pred_groups['Fluxcom_era5l change']
#     + pred_groups['Fluxcom_era5l avg']
#     # + pred_groups['WorldClim'][2015]
#     # + pred_groups['AlphaEarth 2017+2018'][2015]
#     + pred_groups['doy change']
#     + pred_groups['doy avg']
#     + pred_groups['OC (log) avg'] + pred_groups['LUCAS log avg']
# )
# # Same time-aggregated predictors for hybrid steady-state (replaces single-year 2015/2018 columns).
# predictors_static = (
#     pred_groups['Texture']
#     + pred_groups['Mineral Activity'][2015]
#     + pred_groups['Ox. ex. Al/Fe']
#     # + pred_groups['LUCAS normal avg']
#     + pred_groups['Fluxcom_era5l change']
#     + pred_groups['Fluxcom_era5l avg']
#     # + pred_groups['WorldClim'][2015]
#     # + pred_groups['AlphaEarth 2017+2018'][2015]
#     + pred_groups['doy change']
#     + pred_groups['doy avg']
#     # + pred_groups['OC (log) avg'] + pred_groups['LUCAS log avg']
# )
# log_cols = pred_groups['LUCAS log'][2015] + pred_groups['LUCAS log'][2018]+ pred_groups['OC (log) avg'] + pred_groups['LUCAS log avg']

default_param_ranges = {
    "I": {
        "min": 0.05,
        "default": 1.4,
        "max": 2.0,
        "unit": "mg C g^-1 soil yr^-1", 
        "description": "Carbon input rate (enters particulate SOC pool)"
    },
    "CUE": {
        "min": 0.1,
        "default": 0.47,
        "max": 0.8,
        "unit": "-",
        "description": "Microbial carbon use efficiency"
    },
    "beta": {
        "min": 0.5,
        "default": 1.0,
        "max": 2.5,
        "unit": "-",
        "description": "Density‑dependence exponent for microbial turnover"
    },
    "tmb": {
        "min": 0.1,
        "default": "-",
        "max": 1.0,
        "unit": "-",
        "description": "Proportion of microbial turnover transferred to mineral‑associated SOC"
    },
    "Cg0b": {
        "min": 0.0005,
        "default": 2.0,
        "max": 10.0,
        "unit": "mg C g^-1 soil",
        "description": "Microbial biomass at which microbial growth rate becomes zero"
    },
    "Cg0m": {
        "min": 0.5,
        "default": 27.0,
        "max": 150,
        "unit": "mg C g^-1 soil",
        "description": "Mineral‑associated SOC pool size at which growth rate of that pool is zero"
    },
    "qx": {
        "min": 0.1,
        "default": 1.0,
        "max": 10.0,
        "unit": "-",
        "description": "Dimensionless scaling factor (not used in the model)"
    },
    "Vmax_p": {
        "min": 88 * 0.1,
        "default": 88.0,
        "max": 88 * 10.0,
        "unit": "yr^-1",
        "description": "Maximum decomposition rate for particulate SOC"
    },
    "Vmax_m": {
        "min": 171 * 0.1,
        "default": 171.0,
        "max": 171 * 10.0,
        "unit": "yr^-1",
        "description": "Maximum decomposition rate for mineral‑associated SOC"
    },
    "Km_p": {
        "min": 144 * 0.1,
        "default": 144.0,
        "max": 144 * 10.0,
        "unit": "mg C g^-1 soil",
        "description": "Half‑saturation constant for particulate SOC decomposition"
    },
    "Km_m": {
        "min": 936 * 0.1,
        "default": 936.0,
        "max": 936 * 10.0,
        "unit": "mg C g^-1 soil",
        "description": "Half‑saturation constant for mineral‑associated SOC decomposition"
    },
    "kp": {
        "min": 0.3 * 0.1,
        "default": 0.3,
        "max": 0.3 * 10.0,
        "unit": "yr^-1",
        "description": "First‑order decay rate for particulate SOC"
    },
    "kb": {
        "min": 2 * 0.01,
        "default": 2.5,
        "max": 2 * 1.0,
        "unit": "yr^-1",
        "description": "Microbial turnover rate"
    },
    "km": {
        "min": 0.09 * 0.1,
        "default": 0.09,
        "max": 0.09 * 10.0,
        "unit": "yr^-1",
        "description": "First‑order decay rate for mineral‑associated SOC"
    }
}

default_state_bounds = {
    "Cp": {
        "min": 1.0,
        "default": "-",
        "max": 200.0,
        "unit": "mg C g^-1 soil",
        "description": "Particulate SOC – plant‑derived SOC in a minimally processed state"
    },
    "Cb": {
        "min": 0.00005,
        "default": "-",
        "max": 6.0,
        "unit": "mg C g^-1 soil",
        "description": "Microbial biomass carbon"
    },
    "Cm": {
        "min": 0.05,
        "default": "-",
        "max": 80.0,
        "unit": "mg C g^-1 soil",
        "description": "Mineral‑associated SOC"
    }
}