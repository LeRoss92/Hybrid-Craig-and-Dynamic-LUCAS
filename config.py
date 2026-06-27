# Predictor groups
pred_groups = {
    'Texture': {
        'color': '#f5f5dc',  # Beige
        'average': {
            'Clay': 'Clay',
            'Silt': 'Silt', 
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
        'average': {
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
            'N': 'N_2009', 
            'P': 'P_2009', 
            'K': 'K_2009', 
        },
        '2015': {
            'N': 'N_2015', 
            'P': 'P_2015', 
            'K': 'K_2015', 
        },
        '2018': {
            'N': 'N_2018', 
            'P': 'P_2018', 
            'K': 'K_2018', 
        },
        'average': {
            'N': 'N_avg_09_15_18', 
            'P': 'P_avg_09_15_18', 
            'K': 'K_avg_09_15_18', 
        },
        'change': {
            'N': 'N_linreg_slope', 
            'P': 'P_linreg_slope', 
            'K': 'K_linreg_slope', 
        },
    },
    'Acidity': {
        'color': 'orange',
        '2009': {
            'pH CaCl2': 'pH_CaCl2_2009',
            'pH H2O': 'pH_H2O_2009',
        },
        '2015': {
            'pH CaCl2': 'pH_CaCl2_2015',
            'pH H2O': 'pH_H2O_2015',
        },
        '2018': {
            'pH CaCl2': 'pH_CaCl2_2018',
            'pH H2O': 'pH_H2O_2018',
        },
        'average': {
            'pH CaCl2': 'pH_CaCl2_avg_09_15_18',
            'pH H2O': 'pH_H2O_avg_09_15_18',
        },
        'change': {
            'pH CaCl2': 'pH_CaCl2_linreg_slope',
            'pH H2O': 'pH_H2O_linreg_slope',
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
            'TP': 'era5_land_tp_2009-5_mean',
            'PET': 'era5_land_hpet_2009-5_mean',
            'AI': 'aridity_index_2009-5_mean',
            'WAI1': 'WAI1_2009-5_mean',
            'WAI2': 'WAI2_2009-5_mean',
            'N_dep': 'N_dep_2009-5_mean',
        },
        '2015': {
            'H': 'Fluxcom_H_2015-5_mean',
            'LE': 'Fluxcom_LE_2015-5_mean',
            'T': 'era5_land_t2m_2015-5_mean',
            'TP': 'era5_land_tp_2015-5_mean',
            'PET': 'era5_land_hpet_2015-5_mean',
            'AI': 'aridity_index_2015-5_mean',
            'WAI1': 'WAI1_2015-5_mean',
            'WAI2': 'WAI2_2015-5_mean',
            'N_dep': 'N_dep_2015-5_mean',
        },
        '2018': {
            'H': 'Fluxcom_H_2018-5_mean',
            'LE': 'Fluxcom_LE_2018-5_mean',
            'T': 'era5_land_t2m_2018-5_mean',
            'TP': 'era5_land_tp_2018-5_mean',
            'PET': 'era5_land_hpet_2018-5_mean',
            'AI': 'aridity_index_2018-5_mean',
            'WAI1': 'WAI1_2018-5_mean',
            'WAI2': 'WAI2_2018-5_mean',
            'N_dep': 'N_dep_2018-5_mean',
        },
        'average': {
            'H': 'Fluxcom_H_avg_09_15_18',
            'LE': 'Fluxcom_LE_avg_09_15_18',
            'T': 'era5_land_t2m_avg_09_15_18',
            'TP': 'era5_land_tp_avg_09_15_18',
            'PET': 'era5_land_hpet_avg_09_15_18',
            'AI': 'aridity_index_avg_09_15_18',
            'WAI1': 'WAI1_avg_09_15_18',
            'WAI2': 'WAI2_avg_09_15_18',
            'N_dep': 'N_dep_avg_09_15_18',
        },
        'change': {
            'H': 'Fluxcom_H_linreg_slope',
            'LE': 'Fluxcom_LE_linreg_slope',
            'T': 'era5_land_t2m_linreg_slope',
            'TP': 'era5_land_tp_linreg_slope',
            'PET': 'era5_land_hpet_linreg_slope',
            'AI': 'aridity_index_linreg_slope',
            'WAI1': 'WAI1_linreg_slope',
            'WAI2': 'WAI2_linreg_slope',
            'N_dep': 'N_dep_linreg_slope',
        },
    },
    # 'Climate': {
    #     'color': '#ff9999',  # Light red
    #     'average': {
    #         'MAT':                                  'Chelsea_2015gps_Bio01',
    #         'Diurnal Range':                        'Chelsea_2015gps_Bio02',
    #         'Isothermality':                        'Chelsea_2015gps_Bio03',
    #         'Temperature Seasonality':              'Chelsea_2015gps_Bio04',
    #         'Max Temperature of Warmest Month':     'Chelsea_2015gps_Bio05',
    #         'Min Temperature of Coldest Month':     'Chelsea_2015gps_Bio06',
    #         'Temperature Annual Range':             'Chelsea_2015gps_Bio07',
    #         'Mean Temperature of Wettest Quarter':  'Chelsea_2015gps_Bio08',
    #         'Mean Temperature of Driest Quarter':   'Chelsea_2015gps_Bio09',
    #         'Mean Temperature of Warmest Quarter':  'Chelsea_2015gps_Bio10',
    #         'Mean Temperature of Coldest Quarter':  'Chelsea_2015gps_Bio11',
    #         'Annual Precipitation':                 'Chelsea_2015gps_Bio12',
    #         'Precipitation of Wettest Month':       'Chelsea_2015gps_Bio13',
    #         'Precipitation of Driest Month':        'Chelsea_2015gps_Bio14',
    #         'Precipitation Seasonality':            'Chelsea_2015gps_Bio15',
    #         'Precipitation of Wettest Quarter':     'Chelsea_2015gps_Bio16',
    #         'Precipitation of Driest Quarter':      'Chelsea_2015gps_Bio17',
    #         'Precipitation of Warmest Quarter':     'Chelsea_2015gps_Bio18',
    #         'Precipitation of Coldest Quarter':     'Chelsea_2015gps_Bio19',
    #     },
    # },
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
    'Elevation': {
        'color': 'grey',
        'average': {
            'Elevation': 'gps_altitude_avg_09_15_18',
        },
    },
    'Water': {
        'color': 'grey',
        'average': {
            'Water Table': 'WT_2015gps_average',
        },
    },
    'Extremes': {
        'color': 'brown',
        '2009': {
            'Summer Days': 'SU_2009-5_mean',
            'Flooded': 'Flood_2009-5_mean',
            'Frost Days': 'FD_2009-5_mean',
            'Consecutive Dry Days': 'CDD_2009-5_mean',
        },
        '2015': {
            'Summer Days': 'SU_2015-5_mean',
            'Flooded': 'Flood_2015-5_mean',
            'Frost Days': 'FD_2015-5_mean',
            'Consecutive Dry Days': 'CDD_2015-5_mean',
        },
        '2018': {
            'Summer Days': 'SU_2018-5_mean',
            'Flooded': 'Flood_2018-5_mean',
            'Frost Days': 'FD_2018-5_mean',
            'Consecutive Dry Days': 'CDD_2018-5_mean',
        },
        'average': {
            'Summer Days': 'SU_avg_09_15_18',
            'Flooded': 'Flood_avg_09_15_18',
            'Frost Days': 'FD_avg_09_15_18',
            'Consecutive Dry Days': 'CDD_avg_09_15_18',
        },
        'change': {
            'Summer Days': 'SU_linreg_slope',
            'Flooded': 'Flood_linreg_slope',
            'Frost Days': 'FD_linreg_slope',
            'Consecutive Dry Days': 'CDD_linreg_slope',
        },
    },
    'Erosion': {
        'color': 'grey',
        'average': {
            'Wind Erosion': 'Wind_Erosion_2015gps_average',
            'Water Erosion': 'Water_Erosion_2015gps_average',
        },
    },
}
to_log = ['N', 'Clay', 'Coarse', 'Stones', 'Al', 'Fe', 'SOC'] 
to_2log = ['P', 'K']

# Single dict: targets at first level. Per target: predictors, log_predictors, categoricals, inference.
# inference = list of {target_name, predictors, log_predictors, categoricals} - each item is a full pred_config
TARGET_CONFIG = {
    "MAOC": {
        "target_name": "MAOC_index_2009",
        "predictor_groups": [
            ("Texture", "average"),
            # ("Texture", "2009"),
            ("Bulk Density", "average"),
            ("Mineralogy", "average"),
            ("Nutrients", "2009"),
            ("Acidity", "2009"),
            # ("Season", "2009"),
            # ("LC", "2009"),
            ("Atmosphere", "average"),
            ("NPP", "2009"),
            ("Elevation", "average"),
            ("Water", "average"),
            ("Extremes", "2009"),
            ("Erosion", "average"),
            ],
    },
    "MIC": {
        "target_name": "Cmic_index_2018",
        "predictor_groups": [
            ("Texture", "average"),
            ("Texture", "2018"),
            ("Bulk Density", "average"),
            ("Mineralogy", "average"),
            ("Nutrients", "2018"),
            ("Acidity", "2018"),
            ("Season", "2018"),
            # ("LC", "2018"),
            ("Atmosphere", "average"),
            ("NPP", "2018"),
            ("Elevation", "average"),
            ("Water", "average"),
            ("Extremes", "2018"),
            ("Erosion", "average"),
            ],
    },
    "dSOC": {
        "target_name": "OC_linreg_slope",
        "predictor_groups": [
            ("Texture", "average"),
            ("Texture", "change"),
            ("Bulk Density", "average"),
            ("Mineralogy", "average"),
            ("Nutrients", "average"),
            # ("Nutrients", "change"),
            ("Acidity", "average"),
            # ("Acidity", "change"),
            ("Season", "average"),
            ("Season", "change"),
            ("LC", "average"),
            ("LC", "change"),
            ("Atmosphere", "average"),
            # ("Climate", "average"),
            ("Atmosphere", "change"),
            ("NPP", "average"),
            ("NPP", "change"),
            ("Elevation", "average"),
            ("Water", "average"),
            ("Extremes", "average"),
            ("Extremes", "change"),
            ],
    },
    "SOC": {
        "target_name": "OC_avg_09_15_18",
        "predictor_groups": [
            ("Texture", "average"),
            ("Bulk Density", "average"),
            ("Mineralogy", "average"),
            ("Nutrients", "average"),
            ("Acidity", "average"),
            ("Season", "average"),
            ("LC", "average"),
            ("Atmosphere", "average"),
            # ("Climate", "average"),
            ("NPP", "average"),
            ("Elevation", "average"),
            ("Water", "average"),
            ("Extremes", "average"),
            ],
    }
}

TRAIN_DEFAULTS = {
    'models': ['LinReg', 'XGB-1', 'XGB-n'], # , 'Piecewise_Linear_Reg'
    'seed': 42,
    'max_features': 20,
    'n_folds_HP_opt': 3, 
    'n_jobs_folds': 8,
    'N_JOBS': 70
}

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

default_Q10_ranges = {
    "Vmax_p": {
        "min": 1.0,
        "default": 1.4,
        "max": 3.0,
        "unit": "-", 
        "description": "-"
    },
    "Vmax_m": {
        "min": 1.0,
        "default": 1.4,
        "max": 3.0,
        "unit": "-", 
        "description": "-"
    },
    "Km_p": {
        "min": 1.0,
        "default": 1.4,
        "max": 3.0,
        "unit": "-", 
        "description": "-"
    },
    "Km_m": {
        "min": 1.0,
        "default": 1.4,
        "max": 3.0,
        "unit": "-", 
        "description": "-"
    },
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