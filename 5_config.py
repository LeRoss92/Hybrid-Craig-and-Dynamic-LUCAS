# Global parameters
models = ['LinReg', 'Piecewise_Linear_Reg', 'DecisionTree']
n_folds_outter = 10
max_features = 15
n_folds_HP_opt = 3

seed = 42

# Training configs
MAOC_config = {
        'target_name': 'OC_sc_g_kg_2009',
        'target_log': False,
        'predictors': {
            "normal": (
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2009' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2009gps_{year}' for year in [2007, 2008, 2009]]
                + [f'BIOCLIM_2009gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2009' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
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
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
                + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2018' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }

dOC_config = {
        'target_name': 'dOC_15_18',
        'target_log': False,
        'predictors': {
            "normal": (
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2015gps_{year}' for year in [2016, 2017, 2018]]
                + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }

dMAOC_config = {
        'target_name': 'dOC_sc_g_kg_15_18_median',
        'target_log': False,
        'predictors': {
            "normal": (
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2015gps_{year}' for year in [2016, 2017, 2018]]
                + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }

dMIC_config = {
        'target_name': 'dCmic_15_18_median',
        'target_log': False,
        'predictors': {
            "normal": (
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2015gps_{year}' for year in [2016, 2017, 2018]]
                + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
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
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
                + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2018' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }

OC_2015_config = {
        'target_name': 'OC_2015',
        'target_log': False,
        'predictors': {
            "normal": (
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2015gps_{year}' for year in [2013, 2014, 2015]]
                + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2015' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }

OC_2018_config = {
        'target_name': 'OC_2018',
        'target_log': False,
        'predictors': {
            "normal": (
                ['Clay', 'Silt', 'Sand', 'Coarse', ]
                + ['Ox_Al_2018', 'Ox_Fe_2018']
                + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
                + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
                + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
                + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
                + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
            ),
            "log": (
                []
                + [x + '_2018' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
            )
        },
        'models': models,
        'n_folds_outter': n_folds_outter,
        'seed': seed,
        'max_features': max_features,
        'n_folds_HP_opt': n_folds_HP_opt,
        'n_jobs_folds': 32  # Parallelize folds (one fold per worker)
    }




# OC_sc_g_kg prediction configs
pred_config_OC_2009 = {
    'target_name': 'OC_sc_g_kg_2009',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2009' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2009gps_{year}' for year in [2007, 2008, 2009]]
            + [f'BIOCLIM_2009gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2009' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_OC_2015 = {
    'target_name': 'OC_sc_g_kg_2015',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2013, 2014, 2015]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_OC_2018 = {
    'target_name': 'OC_sc_g_kg_2018',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2018' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

# Cmic prediction configs
pred_config_Cmic_2009 = {
    'target_name': 'Cmic_2009',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2009' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2009gps_{year}' for year in [2007, 2008, 2009]]
            + [f'BIOCLIM_2009gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2009' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_Cmic_2015 = {
    'target_name': 'Cmic_2015',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2013, 2014, 2015]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_Cmic_2018 = {
    'target_name': 'Cmic_2018',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2018' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

# Delta prediction configs
pred_config_dOC_15_18 = {
    'target_name': 'dOC_15_18',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_dOC_sc_g_kg_15_18_median = {
    'target_name': 'dOC_sc_g_kg_15_18_median',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_dCmic_15_18_median = {
    'target_name': 'dCmic_15_18_median',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

# OC_2015 prediction configs (using different predictor years)
pred_config_OC_2015_2009 = {
    'target_name': 'OC_2015_2009',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2009' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2009gps_{year}' for year in [2007, 2008, 2009]]
            + [f'BIOCLIM_2009gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2009' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_OC_2015_2015 = {
    'target_name': 'OC_2015_2015',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2013, 2014, 2015]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_OC_2015_2018 = {
    'target_name': 'OC_2015_2018',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2018' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

# OC_2018 prediction configs (using different predictor years)
pred_config_OC_2018_2009 = {
    'target_name': 'OC_2018_2009',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2009' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2009gps_{year}' for year in [2007, 2008, 2009]]
            + [f'BIOCLIM_2009gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2009' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_OC_2018_2015 = {
    'target_name': 'OC_2018_2015',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2013, 2014, 2015]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_OC_2018_2018 = {
    'target_name': 'OC_2018_2018',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2018' for x in ['N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

# BD prediction configs
pred_config_BD_2009 = {
    'target_name': 'BD 0-20_2009',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2009' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2009gps_{year}' for year in [2007, 2008, 2009]]
            + [f'BIOCLIM_2009gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2009' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_BD_2015 = {
    'target_name': 'BD 0-20_2015',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2015' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2015gps_{year}' for year in [2013, 2014, 2015]]
            + [f'BIOCLIM_2015gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2015' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

pred_config_BD_2018 = {
    'target_name': 'BD 0-20_2018',
    'target_log': False,
    'predictors': {
        'normal': (
            ['Clay', 'Silt', 'Sand', 'Coarse', ]
            + ['Ox_Al_2018', 'Ox_Fe_2018']
            + [x + '_2018' for x in ['Calc', 'pH_c', 'pH', ]]
            + [f'MODIS_NPP_2018gps_{year}' for year in [2016, 2017, 2018]]
            + [f'BIOCLIM_2018gps_{i}' for i in range(1, 20)]
            + [f'AE{str(i).zfill(2)}_2018gps_2017' for i in range(64)]
            + [f'AE{str(i).zfill(2)}_2018gps_2018' for i in range(64)]
        ),
        'log': (
            []
            + [x + '_2018' for x in ['OC', 'N', 'P', 'K', 'CaCO3_c', 'CNr', 'CPr', 'CKr', ]]
        )
    }
}

