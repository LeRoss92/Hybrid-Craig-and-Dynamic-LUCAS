# Global parameters
models = ['LinReg', 'Piecewise_Linear_Reg', 'DecisionTree']
n_folds_outter = 10
max_features = 15
n_folds_HP_opt = 3

seed = 42

pred_groups = {
    'Texture': ['Clay', 'Silt', 'Sand', 'Coarse', ],
    'Mineral Activity': {
        2009: ['HALA_2009gps_topsoil'],
        2015: ['HALA_2015gps_topsoil'],
        2018: ['HALA_2018gps_topsoil'],
    },
    'Ox. ex. Al/Fe': ['Ox_Al_2018', 'Ox_Fe_2018'],
    'MODIS NPP 20xx-2a': {
        2009: [f'MODIS_NPP_2009gps_{year}' for year in [2007, 2008, 2009]],
        2015: [f'MODIS_NPP_2015gps_{year}' for year in [2013, 2014, 2015]],
        2018: [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]],
    },
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
    'LUCAS normal dist.': {
        2009: [x + '_2009' for x in ['Calc', 'pH_c', 'pH', ]],
        2015: [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]],
        2018: [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]],
    },
    'LUCAS log dist.': {
        2009: [x + '_2009' for x in ['N', 'P', 'K', 'CaCO3_c']],
        2015: [x + '_2015' for x in ['N', 'P', 'K', 'CaCO3_c']],
        2018: [x + '_2018' for x in ['N', 'P', 'K', 'CaCO3_c']],
    },
    'OC (log)': {
        2009: ['OC_2009'],
        2015: ['OC_2015'],
        2018: ['OC_2018'],
    },
}

# Training configs
MAOC_config = {
        'target_name': 'OC_sc_g_kg_2009',
        'target_log': False,
        'predictors': {
            "normal": (
                pred_groups['Texture']
                + pred_groups['Mineral Activity'][2009]
                + pred_groups['Ox. ex. Al/Fe']
                + pred_groups['LUCAS normal dist.'][2009]
                + pred_groups['MODIS NPP 20xx-2a'][2009]
                + pred_groups['WorldClim'][2009]
                + pred_groups['AlphaEarth 2017+2018'][2009]
            ),
            "log": (pred_groups['OC (log)'][2009] + pred_groups['LUCAS log dist.'][2009])
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }
MIC_config = {
        'target_name': 'Cmic_2018',
        'target_log': False,
        'predictors': {
            "normal": (
                pred_groups['Texture']
                + pred_groups['Mineral Activity'][2018]
                + pred_groups['Ox. ex. Al/Fe']
                + pred_groups['LUCAS normal dist.'][2018]
                + pred_groups['MODIS NPP 20xx-2a'][2018]
                + pred_groups['WorldClim'][2018]
                + pred_groups['AlphaEarth 2017+2018'][2018]
            ),
            "log": (pred_groups['OC (log)'][2018] + pred_groups['LUCAS log dist.'][2018])
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }
BD_config = {
        'target_name': 'BD 0-20_2018',
        'target_log': False,
        'predictors': {
            "normal": (
                pred_groups['Texture']
                + pred_groups['Mineral Activity'][2018]
                + pred_groups['Ox. ex. Al/Fe']
                + pred_groups['LUCAS normal dist.'][2018]
                + pred_groups['MODIS NPP 20xx-2a'][2018]
                + pred_groups['WorldClim'][2018]
                + pred_groups['AlphaEarth 2017+2018'][2018]
            ),
            "log": (pred_groups['OC (log)'][2018] + pred_groups['LUCAS log dist.'][2018])
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }

# prediction configs
pred_config_MAOC_2015 = {
    'target_name': 'OC_sc_g_kg_2015',
    'target_log': False,
    'predictors': {
        "normal": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2015]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal dist.'][2015]
            + pred_groups['MODIS NPP 20xx-2a'][2015]
            + pred_groups['WorldClim'][2015]
            + pred_groups['AlphaEarth 2017+2018'][2015]
        ),
        "log": (pred_groups['OC (log)'][2015] + pred_groups['LUCAS log dist.'][2015])
    },
    }
pred_config_MAOC_2018 = {
    'target_name': 'OC_sc_g_kg_2018',
    'target_log': False,
    'predictors': {
        "normal": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2018]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal dist.'][2018]
            + pred_groups['MODIS NPP 20xx-2a'][2018]
            + pred_groups['WorldClim'][2018]
            + pred_groups['AlphaEarth 2017+2018'][2018]
        ),
        "log": (pred_groups['OC (log)'][2018] + pred_groups['LUCAS log dist.'][2018])
    },
    }
pred_config_Cmic_2015 = {
    'target_name': 'Cmic_2015',
    'target_log': False,
    'predictors': {
        "normal": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2015]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal dist.'][2015]
            + pred_groups['MODIS NPP 20xx-2a'][2015]
            + pred_groups['WorldClim'][2015]
            + pred_groups['AlphaEarth 2017+2018'][2015]
        ),
        "log": (pred_groups['OC (log)'][2015] + pred_groups['LUCAS log dist.'][2015])
    },
    }
pred_config_BD_2015 = {
    'target_name': 'BD 0-20_2015',
    'target_log': False,
    'predictors': {
        "normal": (
            pred_groups['Texture']
            + pred_groups['Mineral Activity'][2015]
            + pred_groups['Ox. ex. Al/Fe']
            + pred_groups['LUCAS normal dist.'][2015]
            + pred_groups['MODIS NPP 20xx-2a'][2015]
            + pred_groups['WorldClim'][2015]
            + pred_groups['AlphaEarth 2017+2018'][2015]
        ),
        "log": (pred_groups['OC (log)'][2015] + pred_groups['LUCAS log dist.'][2015])
    },
    }

default_param_ranges = {
    "I": {"min": 0.05, "max": 2.0},                             # "I": {"min": 0.0, "max": 0.5},
    "CUE": {"min": 0.1, "max": 0.8},                            # "CUE": {"min": 0.2, "max": 0.8},
    "beta": {"min": 0.5, "max": 2.5},                           # "beta": {"min": 0.8, "max": 2.0},
    "tmb": {"min": 0.1, "max": 1.0},                            # "tmb": {"min": 0.3, "max": 1.2},
    "Cg0b": {"min": 0.0005, "max": 10.0},                       # "Cg0b": {"min": 5.0, "max": 20.0},
    "Cg0m": {"min": 0.5 * 1.3, "max": 150 * 1.3},               # "Cg0m": {"min": 5.0, "max": 20.0},
    "qx": {"min": 0.1, "max": 10.0},                            # "qx": {"min": 0.5, "max": 2.0},
    "Vmax_p": {"min": 88 * 0.1, "max": 88 * 10.0},              # "Vmax_p": {"min": 0.1, "max": 1.0},
    "Vmax_m": {"min": 171 * 0.1, "max": 171 * 10.0},            # "Vmax_m": {"min": 0.1, "max": 1.0},
    "Km_p": {"min": 144 * 0.1 * 1.3, "max": 144 * 10.0 * 1.3},  # "Km_p": {"min": 0.5, "max": 2.0},
    "Km_m": {"min": 936 * 0.1 * 1.3, "max": 936 * 10.0 * 1.3},  # "Km_m": {"min": 0.5, "max": 2.0},
    "kp": {"min": 0.3 * 0.1, "max": 0.3 * 10.0},                # "kp": {"min": 0.01, "max": 0.2},
    "kb": {"min": 2 * 0.01, "max": 2 * 1.0},                    # "kb": {"min": 0.01, "max": 0.2},
    "km": {"min": 0.09 * 0.1, "max": 0.09 * 10.0},              # "km": {"min": 0.01, "max": 0.2},
    }
default_state_bounds = {
    "Cp": {"min": 1.0, "max": 200.0},                           # "Cp": {"min": 0.1, "max": 2.0},
    "Cb": {"min": 0.00005, "max": 6.0},                         # "Cb": {"min": 0.1, "max": 2.0},
    "Cm": {"min": 0.05, "max": 80.0},                           # "Cm": {"min": 0.1, "max": 2.0},
    }








predictors_dynamic = (
    pred_groups['Texture']
    + pred_groups['Ox. ex. Al/Fe']
    + pred_groups['Mineral Activity'][2015]
    + pred_groups['MODIS NPP 20xx-2a'][2015]
    + pred_groups['WorldClim'][2015]
    + pred_groups['LUCAS normal dist.'][2015]
    + pred_groups['AlphaEarth 2017+2018'][2015]
    + pred_groups['LUCAS log dist.'][2015]
    )
predictors_2015 = (
    pred_groups['Texture']
    + pred_groups['Ox. ex. Al/Fe']
    + pred_groups['Mineral Activity'][2015]
    + pred_groups['MODIS NPP 20xx-2a'][2015]
    + pred_groups['WorldClim'][2015]
    + pred_groups['LUCAS normal dist.'][2015]
    + pred_groups['AlphaEarth 2017+2018'][2015]
    + pred_groups['LUCAS log dist.'][2015]
    )
predictors_2018 = (
    pred_groups['Texture']
    + pred_groups['Ox. ex. Al/Fe']
    + pred_groups['Mineral Activity'][2018]
    + pred_groups['MODIS NPP 20xx-2a'][2018]
    + pred_groups['WorldClim'][2018]
    + pred_groups['LUCAS normal dist.'][2018]
    + pred_groups['AlphaEarth 2017+2018'][2018]
    + pred_groups['LUCAS log dist.'][2018]
    )

log_cols = pred_groups['LUCAS log dist.'][2015] + pred_groups['LUCAS log dist.'][2018]
