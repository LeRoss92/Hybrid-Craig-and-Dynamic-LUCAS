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
    ["Clay", "Silt", "Sand", "Coarse"]
    + ["Ox_Al_2018", "Ox_Fe_2018"]
    + [x + "_2015" for x in ["Calc", "pH_c", "pH"]]
    + [x + "_2015" for x in ["CaCO3_c"]]  # logged
    + [f"MODIS_NPP_2015gps_{y}" for y in [2015, 2015, 2015]]
    + [f"BIOCLIM_2015gps_{i}" for i in range(1, 20)]
    + [x + "_2015" for x in ["N", "P", "K", "CNr", "CPr", "CKr"]]  # logged
    + [f"AE{str(i).zfill(2)}_2018gps_2017" for i in range(64)]
    + [f"AE{str(i).zfill(2)}_2018gps_2018" for i in range(64)]
)

predictors_2015 = (
    ["Clay_2015", "Silt_2015", "Sand_2015", "Coarse_2015"]
    + ["Ox_Al_2018", "Ox_Fe_2018"]
    + [x + "_2015" for x in ["Calc", "pH_c", "pH"]]
    + [x + "_2015" for x in ["CaCO3_c"]]  # logged
    + [f"MODIS_NPP_2015gps_{y}" for y in [2015, 2015, 2015]]
    + [f"BIOCLIM_2015gps_{i}" for i in range(1, 20)]
    + [x + "_2015" for x in ["N", "P", "K", "CNr", "CPr", "CKr"]]  # logged
    + [f"AE{str(i).zfill(2)}_2018gps_2017" for i in range(64)]
    + [f"AE{str(i).zfill(2)}_2018gps_2018" for i in range(64)]
)

predictors_2018 = (
    ["Clay", "Silt", "Sand", "Coarse"]
    + ["Ox_Al_2018", "Ox_Fe_2018"]
    + [x + "_2018" for x in ["Calc", "pH_c", "pH"]]
    + [x + "_2018" for x in ["CaCO3_c"]]  # logged
    + [f"MODIS_NPP_2018gps_{y}" for y in [2018, 2018, 2018]]
    + [f"BIOCLIM_2018gps_{i}" for i in range(1, 20)]
    + [x + "_2018" for x in ["N", "P", "K", "CNr", "CPr", "CKr"]]  # logged
    + [f"AE{str(i).zfill(2)}_2018gps_2017" for i in range(64)]
    + [f"AE{str(i).zfill(2)}_2018gps_2018" for i in range(64)]
)

log_cols = [
    "CaCO3_c_2015",
    "N_2015",
    "P_2015",
    "K_2015",
    "CNr_2015",
    "CPr_2015",
    "CKr_2015",
    "CaCO3_c_2018",
    "N_2018",
    "P_2018",
    "K_2018",
    "CNr_2018",
    "CPr_2018",
    "CKr_2018",
]
