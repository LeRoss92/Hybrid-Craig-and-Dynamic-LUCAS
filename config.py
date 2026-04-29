# Predictor groups (columns discovered from df)
pred_groups = {
    'Texture': ['Clay', 'Silt', 'Sand', 'Coarse'],
    'Mineral Activity': {
        2009: ['HALA_2009gps_topsoil'],
        2015: ['HALA_2015gps_topsoil'],
        2018: ['HALA_2018gps_topsoil'],
    },
    'Ox. ex. Al/Fe': ['Ox_Al_2018', 'Ox_Fe_2018'],
    'AlphaEarth 2017+2018': {
        2009: [f'AE{str(i).zfill(2)}_2009gps_2017' for i in range(64)] + [f'AE{str(i).zfill(2)}_2009gps_2018' for i in range(64)],
        2015: [f'AE{str(i).zfill(2)}_2015gps_2017' for i in range(64)] + [f'AE{str(i).zfill(2)}_2015gps_2018' for i in range(64)],
        2018: [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)] + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)],
    },
    'WorldClim': {
        2009: [f'BIOCLIM_2009gps_{i}' for i in range(1, 20)],
        2015: [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)],
        2018: [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)],
    },
    'LUCAS normal': {
        2009: [x + '_2009' for x in ['Calc', 'pH_c', 'pH']],
        2015: [x + '_2015' for x in ['Calc', 'pH_c', 'pH']],
        2018: [x + '_2018' for x in ['Calc', 'pH_c', 'pH']],
    },
    'LUCAS log': {
        2009: [x + '_2009' for x in ['N', 'P', 'K', 'CaCO3_c']],
        2015: [x + '_2015' for x in ['N', 'P', 'K', 'CaCO3_c']],
        2018: [x + '_2018' for x in ['N', 'P', 'K', 'CaCO3_c']],
    },
    'OC (log)': {
        2009: ['OC_2009'],
        2015: ['OC_2015'],
        2018: ['OC_2018'],
    },
    'LUCAS normal avg': [x + '_avg_09_15_18' for x in ['Calc', 'pH_c', 'pH']],
    'LUCAS log avg': [x + '_avg_09_15_18' for x in ['N', 'P', 'K', 'CaCO3_c']],
    'OC (log) avg': ['OC_avg_09_15_18'],
    'doy': {
        2009: ['doy_2009'],
        2015: ['doy_2015'],
        2018: ['doy_2018'],
    },
    'doy change': ["doy_linreg_slope"],
    'doy avg': ["doy_avg_09_15_18"],
    'Fluxcom_era5l': {
        2009: ['Fluxcom_H_2009-5_mean',
                'Fluxcom_LE_2009-5_mean', 
                'era5_land_t2m_2009-5_mean', 
                'era5_land_tp_2009-5_mean', 
                'era5_land_hpet_2009-5_mean', 
                'MODIS_NPP_2009-5_mean'],
        2015: ['Fluxcom_H_2009-5_mean',
                'Fluxcom_LE_2009-5_mean', 
                'era5_land_t2m_2009-5_mean', 
                'era5_land_tp_2009-5_mean', 
                'era5_land_hpet_2009-5_mean', 
                'MODIS_NPP_2015-5_mean'],
        2018: ['Fluxcom_H_2009-5_mean',
                'Fluxcom_LE_2009-5_mean', 
                'era5_land_t2m_2009-5_mean', 
                'era5_land_tp_2009-5_mean', 
                'era5_land_hpet_2009-5_mean', 
                'MODIS_NPP_2009-5_mean'],
    },
    'Fluxcom_era5l change': [x + '_linreg_slope' for x in ['Fluxcom_H', 'Fluxcom_LE', 'era5_land_t2m', 'era5_land_tp', 'era5_land_hpet', 'MODIS_NPP']],
    'Fluxcom_era5l avg': [x + '_avg_09_15_18' for x in ['Fluxcom_H', 'Fluxcom_LE', 'era5_land_t2m', 'era5_land_tp', 'era5_land_hpet', 'MODIS_NPP']],
    'input': {
        2009: ['Fluxcom_H_2009-5_mean'
                'Fluxcom_LE_2009-5_mean', 
                'era5_land_t2m_2009-5_mean', 
                'era5_land_tp_2009-5_mean', 
                'era5_land_hpet_2009-5_mean'],
        2015: ['Fluxcom_H_2009-5_mean'
                'Fluxcom_LE_2009-5_mean', 
                'era5_land_t2m_2009-5_mean', 
                'era5_land_tp_2009-5_mean', 
                'era5_land_hpet_2009-5_mean'],
        2018: ['Fluxcom_H_2009-5_mean'
                'Fluxcom_LE_2009-5_mean', 
                'era5_land_t2m_2009-5_mean', 
                'era5_land_tp_2009-5_mean', 
                'era5_land_hpet_2009-5_mean'],
    },
}

# Single dict: targets at first level. Per target: predictors, log_predictors, categoricals, inference.
# inference = list of {target_name, predictors, log_predictors, categoricals} - each item is a full pred_config
TARGET_CONFIG = {
    "MAOC": {
        "target_name": "MAOC_index_2009",
        "target_log": False,
        "predictors": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2009]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal'][2009]
            + pred_groups['Fluxcom_era5l'][2009]
            # + pred_groups['WorldClim'][2009]
            # + pred_groups['AlphaEarth 2017+2018'][2009]
            + pred_groups['doy'][2009]
        ),
        "log_predictors": pred_groups['OC (log)'][2009] + pred_groups['LUCAS log'][2009],
        "categoricals": ['lc1_2_2009'], # , 'lc1_2009' 'Soil_Group', 
        "inference": [
            {
                "target_name": "MAOC_index_2015",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2015]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2015]
                    + pred_groups['Fluxcom_era5l'][2015]
                    # + pred_groups['WorldClim'][2015]
                    # + pred_groups['AlphaEarth 2017+2018'][2015]
                    + pred_groups['doy'][2015]
                ),
                "log_predictors": pred_groups['OC (log)'][2015] + pred_groups['LUCAS log'][2015],
                "categoricals": ['lc1_2_2015'] # , 'lc1_2015' 'Soil_Group', 
            },
            {
                "target_name": "MAOC_index_2018",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2018]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2018]
                    + pred_groups['Fluxcom_era5l'][2018]
                    # + pred_groups['WorldClim'][2018]
                    # + pred_groups['AlphaEarth 2017+2018'][2018]
                    + pred_groups['doy'][2018]
                ),
                "log_predictors": pred_groups['OC (log)'][2018] + pred_groups['LUCAS log'][2018],
                "categoricals": ['lc1_2_2018'] # , 'lc1_2018' 'Soil_Group', 
            },
            {
                "target_name": "MAOC_index_2009",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2009]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2009]
                    + pred_groups['Fluxcom_era5l'][2009]
                    # + pred_groups['WorldClim'][2009]
                    # + pred_groups['AlphaEarth 2017+2018'][2009]
                    + pred_groups['doy'][2009]
                ),
                "log_predictors": pred_groups['OC (log)'][2009] + pred_groups['LUCAS log'][2009],
                "categoricals": ['lc1_2_2009'], # , 'lc1_2009' 'Soil_Group', 
            },
        ],
    },
    "MIC": {
        "target_name": "Cmic_index_2018",
        "target_log": False,
        "predictors": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2018]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal'][2018]
            + pred_groups['Fluxcom_era5l'][2018]
            # + pred_groups['WorldClim'][2018]
            # + pred_groups['AlphaEarth 2017+2018'][2018]
            + pred_groups['doy'][2018]
        ),
        "log_predictors": pred_groups['OC (log)'][2018] + pred_groups['LUCAS log'][2018],
        "categoricals": ['lc1_2_2018'], # , 'lc1_2018' 'Soil_Group', 
        "inference": [
            {
                "target_name": "Cmic_index_2015",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2015]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2015]
                    + pred_groups['Fluxcom_era5l'][2015]
                    # + pred_groups['WorldClim'][2015]
                    # + pred_groups['AlphaEarth 2017+2018'][2015]
                    + pred_groups['doy'][2015]
                ),
                "log_predictors": pred_groups['OC (log)'][2015] + pred_groups['LUCAS log'][2015],
                "categoricals": ['lc1_2_2015'] # , 'lc1_2015' 'Soil_Group', 
            },
            {
                "target_name": "Cmic_index_2018",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2018]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2018]
                    + pred_groups['Fluxcom_era5l'][2018]
                    # + pred_groups['WorldClim'][2018]
                    # + pred_groups['AlphaEarth 2017+2018'][2018]
                    + pred_groups['doy'][2018]
                ),
                "log_predictors": pred_groups['OC (log)'][2018] + pred_groups['LUCAS log'][2018],
                "categoricals": ['lc1_2_2018'], #, 'lc1_2018' 'Soil_Group', 
            },
            {
                "target_name": "Cmic_index_2009",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2009]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2009]
                    + pred_groups['Fluxcom_era5l'][2009]
                    # + pred_groups['WorldClim'][2009]
                    # + pred_groups['AlphaEarth 2017+2018'][2009]
                    + pred_groups['doy'][2009]
                ),
                "log_predictors": pred_groups['OC (log)'][2009] + pred_groups['LUCAS log'][2009],
                "categoricals": ['lc1_2_2009'], #, 'lc1_2009' 'Soil_Group', 
            },
        ],
    },
    "SOC09": {
        "target_name": "OC_2009",
        "target_log": False,
        "predictors": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2009]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal'][2009]
            + pred_groups['Fluxcom_era5l'][2009]
            # + pred_groups['WorldClim'][2009]
            # + pred_groups['AlphaEarth 2017+2018'][2009]
            + pred_groups['doy'][2009]
        ),
        "log_predictors": pred_groups['LUCAS log'][2009],
        "categoricals": ['lc1_2_2009'], # 'Soil_Group', , 'lc1_2009'
        "inference": [
            {
                "target_name": "OC_2009",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2009]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2009]
                    + pred_groups['Fluxcom_era5l'][2009]
                    # + pred_groups['WorldClim'][2009]
                    # + pred_groups['AlphaEarth 2017+2018'][2009]
                    + pred_groups['doy'][2009]
                ),
                "log_predictors": pred_groups['LUCAS log'][2009],
                "categoricals": ['lc1_2_2009'], # 'Soil_Group', , 'lc1_2009'
            },
            {
                "target_name": "OC_2015",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2015]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2015]
                    + pred_groups['Fluxcom_era5l'][2015]
                    # + pred_groups['WorldClim'][2015]
                    # + pred_groups['AlphaEarth 2017+2018'][2015]
                    + pred_groups['doy'][2015]
                ),
                "log_predictors": pred_groups['LUCAS log'][2015],
                "categoricals": ['lc1_2_2015'], # 'Soil_Group', , 'lc1_2015'
            },
            {
                "target_name": "OC_2018",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2018]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2018]
                    + pred_groups['Fluxcom_era5l'][2018]
                    # + pred_groups['WorldClim'][2018]
                    # + pred_groups['AlphaEarth 2017+2018'][2018]
                    + pred_groups['doy'][2018]
                ),
                "log_predictors": pred_groups['LUCAS log'][2018],
                "categoricals": ['lc1_2_2018'], # 'Soil_Group', , 'lc1_2018'
            },
        ],
    },
    "SOC15": {
        "target_name": "OC_2015",
        "target_log": False,
        "predictors": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2015]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal'][2015]
            + pred_groups['Fluxcom_era5l'][2015]
            # + pred_groups['WorldClim'][2015]
            # + pred_groups['AlphaEarth 2017+2018'][2015]
            + pred_groups['doy'][2015]
        ),
        "log_predictors": pred_groups['LUCAS log'][2015],
        "categoricals": ['lc1_2_2015'], # 'Soil_Group', , 'lc1_2015'
        "inference": [
            {
                "target_name": "OC_2009",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2009]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2009]
                    + pred_groups['Fluxcom_era5l'][2009]
                    # + pred_groups['WorldClim'][2009]
                    # + pred_groups['AlphaEarth 2017+2018'][2009]
                    + pred_groups['doy'][2009]
                ),
                "log_predictors": pred_groups['LUCAS log'][2009],
                "categoricals": ['lc1_2_2009'], # 'Soil_Group', , 'lc1_2009'
            },
            {
                "target_name": "OC_2015",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2015]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2015]
                    + pred_groups['Fluxcom_era5l'][2015]
                    # + pred_groups['WorldClim'][2015]
                    # + pred_groups['AlphaEarth 2017+2018'][2015]
                    + pred_groups['doy'][2015]
                ),
                "log_predictors": pred_groups['LUCAS log'][2015],
                "categoricals": ['lc1_2_2015'], # 'Soil_Group', , 'lc1_2015'
            },
            {
                "target_name": "OC_2018",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2018]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2018]
                    + pred_groups['Fluxcom_era5l'][2018]
                    # + pred_groups['WorldClim'][2018]
                    # + pred_groups['AlphaEarth 2017+2018'][2018]
                    + pred_groups['doy'][2018]
                ),
                "log_predictors": pred_groups['LUCAS log'][2018],
                "categoricals": ['lc1_2_2018'], # 'Soil_Group', , 'lc1_2018'
            },
        ],
    },
    "SOC18": {
        "target_name": "OC_2018",
        "target_log": False,
        "predictors": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2018]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal'][2018]
            + pred_groups['Fluxcom_era5l'][2018]
            # + pred_groups['WorldClim'][2018]
            # + pred_groups['AlphaEarth 2017+2018'][2018]
            + pred_groups['doy'][2018]
        ),
        "log_predictors": pred_groups['LUCAS log'][2018],
        "categoricals": ['lc1_2_2018'], # 'Soil_Group', , 'lc1_2018'
        "inference": [
            {
                "target_name": "OC_2009",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2009]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2009]
                    + pred_groups['Fluxcom_era5l'][2009]
                    # + pred_groups['WorldClim'][2009]
                    # + pred_groups['AlphaEarth 2017+2018'][2009]
                    + pred_groups['doy'][2009]
                ),
                "log_predictors": pred_groups['LUCAS log'][2009],
                "categoricals": ['lc1_2_2009'], # 'Soil_Group', , 'lc1_2009'
            },
            {
                "target_name": "OC_2015",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2015]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2015]
                    + pred_groups['Fluxcom_era5l'][2015]
                    # + pred_groups['WorldClim'][2015]
                    # + pred_groups['AlphaEarth 2017+2018'][2015]
                    + pred_groups['doy'][2015]
                ),
                "log_predictors": pred_groups['LUCAS log'][2015],
                "categoricals": ['lc1_2_2015'], # 'Soil_Group', , 'lc1_2015'
            },
            {
                "target_name": "OC_2018",
                "predictors": (
                    pred_groups['Texture']
                    + pred_groups['Mineral Activity'][2018]
                    + pred_groups['Ox. ex. Al/Fe']
                    + pred_groups['LUCAS normal'][2018]
                    + pred_groups['Fluxcom_era5l'][2018]
                    # + pred_groups['WorldClim'][2018]
                    # + pred_groups['AlphaEarth 2017+2018'][2018]
                    + pred_groups['doy'][2018]
                ),
                "log_predictors": pred_groups['LUCAS log'][2018],
                "categoricals": ['lc1_2_2018'], # 'Soil_Group', , 'lc1_2018'
            },
        ],
    },
    "dSOC_09_18": {
        "target_name": "SOC_linreg_slope",
        "target_log": False,
        "predictors": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2015]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal avg']
            + pred_groups['Fluxcom_era5l change']
            + pred_groups['Fluxcom_era5l avg']
            # + pred_groups['WorldClim'][2015]
            # + pred_groups['AlphaEarth 2017+2018'][2015]
            + pred_groups['doy change']
            + pred_groups['doy avg']
        ),
        "log_predictors": pred_groups['OC (log) avg'] + pred_groups['LUCAS log avg'],
        "categoricals": ['LC_change_09_15_18'], # 'Soil_Group', 
        "inference": [
            {"target_name": "SOC_linreg_slope"},  # same as training: no predictors/log_predictors override
        ],
    },
}

use_model = {
    # 'BD': 'LinReg',
    'MAOC': 'LinReg',
    'MIC': 'LinReg',
    # 'dSOC_15_18': 'XGB',
    'dSOC_09_18': 'XGB',
    'SOC09': 'XGB',
    'SOC15': 'XGB',
    'SOC18': 'XGB',
}

TRAIN_DEFAULTS = {
    'models': ['LinReg', 'XGB', 'Piecewise_Linear_Reg'], # 'XGB'
    'seed': 42,
    'max_features': 20,
    'n_folds_HP_opt': 3, 
    'n_jobs_folds': 8,
    'N_JOBS': 70
}

# Legacy exports for 6_hybrid, 7_analysis, sensitivity_analysis
predictors_dynamic = (
    pred_groups['Texture'] 
    + pred_groups['Ox. ex. Al/Fe']
    + pred_groups['Mineral Activity'][2015] 
    # + pred_groups['MODIS NPP 20xx-2a'][2015]
    + pred_groups['WorldClim'][2015] 
    + pred_groups['LUCAS normal'][2015]
    + pred_groups['AlphaEarth 2017+2018'][2015] 
    + pred_groups['LUCAS log'][2015]
)
predictors_2015 = (
    pred_groups['Texture'] 
    + pred_groups['Ox. ex. Al/Fe']
    + pred_groups['Mineral Activity'][2015] 
    # + pred_groups['MODIS NPP 20xx-2a'][2015]
    + pred_groups['WorldClim'][2015] 
    + pred_groups['LUCAS normal'][2015]
    + pred_groups['AlphaEarth 2017+2018'][2015] 
    + pred_groups['LUCAS log'][2015]
)
predictors_2018 = (
    pred_groups['Texture'] 
    + pred_groups['Ox. ex. Al/Fe']
    + pred_groups['Mineral Activity'][2018] 
    # + pred_groups['MODIS NPP 20xx-2a'][2018]
    + pred_groups['WorldClim'][2018] 
    + pred_groups['LUCAS normal'][2018]
    + pred_groups['AlphaEarth 2017+2018'][2018] 
    + pred_groups['LUCAS log'][2018]
)
log_cols = pred_groups['LUCAS log'][2015] + pred_groups['LUCAS log'][2018]

# BD = 1.3  # g cm^-3 (bulk density, used for unit conversions)
BD = 1.0  # g cm^-3 (bulk density, used for unit conversions) -> now contents instead of stock modeled

default_param_ranges = {
    "I": {
        "min": 0.05 * BD,
        "default": 1.4 * BD,
        "max": 2.0 * BD,
        "unit": "mg C cm^-3 soil yr^-1",
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
        "min": 0.0005 * BD,
        "default": 2.0 * BD,
        "max": 10.0 * BD,
        "unit": "mg C cm^-3 soil",
        "description": "Microbial biomass at which microbial growth rate becomes zero"
    },
    "Cg0m": {
        "min": 0.5 * BD,
        "default": 27.0 * BD,
        "max": 150 * BD,
        "unit": "mg C cm^-3 soil",
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
        "min": 144 * 0.1 * BD,
        "default": 144.0,
        "max": 144 * 10.0 * BD,
        "unit": "mg C cm^-3 soil",
        "description": "Half‑saturation constant for particulate SOC decomposition"
    },
    "Km_m": {
        "min": 936 * 0.1 * BD,
        "default": 936.0,
        "max": 936 * 10.0 * BD,
        "unit": "mg C cm^-3 soil",
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
        "min": 1.0 * BD,
        "default": "-",
        "max": 200.0 * BD,
        "unit": "mg C cm^-3 soil",
        "description": "Particulate SOC – plant‑derived SOC in a minimally processed state"
    },
    "Cb": {
        "min": 0.00005 * BD,
        "default": "-",
        "max": 6.0 * BD,
        "unit": "mg C cm^-3 soil",
        "description": "Microbial biomass carbon"
    },
    "Cm": {
        "min": 0.05 * BD,
        "default": "-",
        "max": 80.0 * BD,
        "unit": "mg C cm^-3 soil",
        "description": "Mineral‑associated SOC"
    }
}