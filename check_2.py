import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import pandas as pd
results_path = '6_hybrid_outputs'
folds = 5
res_2015_2018 = pd.read_csv("figures/res_2015_2018.csv")


versions = {}
for md in ["linear", "MM", "RMM"]:
    for mt in ["density_dependent", "linear"]:
        for sat in ["no", "Langmuir"]:
            file_sub_str = f"md{md}_mt{mt}_sat{sat}"

            # colors by md
            color_map = {
                "linear": "black",   # linear md -> black
                "MM": "violet",      # MM -> violet
                "RMM": "red",        # RMM -> red
            }

            # line width by mt
            linewidth = 1 if mt == "linear" else 3  # linear mt thin, density_dependent thick

            # line style by saturation
            linestyle = ":" if sat == "no" else "-"  # no saturation dotted, Langmuir continuous

            versions[file_sub_str] = {
                "color": color_map[md],
                "texture": linestyle,          # reuse existing 'texture' field for linestyle
                "linewidth": linewidth,
                "legend": f"{md}-{mt}-{sat}",
                "md": md,
                "mt": mt,
                "sat": sat,
            }
### new Qmax
def fine_fraction(Clay, Silt):
    """ % -> % """
    return Clay + Silt
def Qmax_Georgiou(fine_fraction, mineral_type):
    """ % -> gC/kg 
    mineral_type and fine_fraction can be scalars or numpy arrays.
    mineral_type: 1.0 = high-activity (kaolinite/smectite), 0.0 = low-activity (chlorite/illite), or array of these.
    """
    return np.where(
        mineral_type == 1.0,
        fine_fraction * 0.86,
        np.where(
            mineral_type == 0.0,
            fine_fraction * 0.48,
            np.nan # undefined if mineral_type not 0 or 1
        )
    )
def mass_to_volume_based(mass_based, BD):
    """ gC/kg & g/cm3 -> kgC/m3 """
    return mass_based*BD
def Georgiou(Clay, Silt, mineral_type, BD):
    """ output in kgC/m3 """
    fin_fra = fine_fraction(Clay, Silt)
    Qmax_mass_based = Qmax_Georgiou(fin_fra, mineral_type)
    Qmax_volume_based = mass_to_volume_based(Qmax_mass_based, BD)
    return Qmax_volume_based

def mineral_content(OC):
    """ g/kg -> %mass """
    return 1 - OC/1000
def fine_content(mineral_content, fine_fraction):
    """ %mass & %mass -> %mass=g fine particles per 100 g soil """
    return mineral_content*fine_fraction
def Qmax_Hassink(fine_content):
    """ g fine particles per 100 g soil -> mgC/g """
    return 0.37 * fine_content + 4.07
def Hassink(Clay, Silt, OC, BD):
    """ output in kgC/m3 """
    fin_fra = fine_fraction(Clay, Silt)
    min_con = mineral_content(OC)
    fin_con = fine_content(min_con, fin_fra)
    Qmax_mass_based = Qmax_Hassink(fin_con)
    Qmax_volume_based = mass_to_volume_based(Qmax_mass_based, BD)
    return Qmax_volume_based
    
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2*3, 3)) # init plot
all_files = os.listdir(results_path) 

lucas = pd.read_pickle("5_with_predictions.pkl")
npp_year = "2015"
npp_col = f"MODIS_NPP_{npp_year}gps_{npp_year}"
npp_mask = (
    lucas[npp_col].notna()
    & np.isfinite(lucas[npp_col])
    & (lucas[npp_col] > 0)
).to_numpy()
# Keep original row ids so we can align with hybrid outputs (indexed like 6_hybrid.py).
lucas_masked = lucas.loc[npp_mask]
### select Langmuir+ss+MAOC/POC
selected_Qmax_versions = ["mdMM_mtlinear_satLangmuir", "mdlinear_mtdensity_dependent_satLangmuir", "mdlinear_mtlinear_satLangmuir"]
# Distinct colors (global `versions` only colors by md, so both linear-md models were black).
selected_qmax_colors = {
    "mdMM_mtlinear_satLangmuir": "violet",
    "mdlinear_mtdensity_dependent_satLangmuir": "#e67e22",
    "mdlinear_mtlinear_satLangmuir": "#1a7f37",
}
for version in selected_Qmax_versions: # loop over versions
    files = [f for f in all_files # get files
            if version in f
            and f"fold{0}" in f
            and f"{'Cp-Cm'}_" in f
            and f"temp2018_fold" in f
            ]
    files = sorted(files, key=len, reverse=True)
    opt = res_2015_2018[res_2015_2018["version"] == version]["optimal n_sp"].values[0]
    max = res_2015_2018[res_2015_2018["version"] == version]["max n_sp"].values[0] -2 # use max instead of opt to include Qmax
    file = files[opt]
    all_predicted_Qmax = [] 
    all_Hassink = [] 
    all_Georgiou = []
    
    # for fold, file in enumerate(files): # loop over folds
    data = pd.read_pickle(os.path.join(results_path, file)) # load pkl
    # Hybrid may have been run on a different NPP mask than current; keep rows in both.
    keep_idx = data.index[data.index.isin(lucas_masked.index)]
    data = data.loc[keep_idx]
    lucas_aligned = lucas_masked.loc[keep_idx]
    all_predicted_Qmax += list(data[['param_Cg0m']].values)
    Clay = lucas_aligned[['Clay']].values
    Silt = lucas_aligned[['Silt']].values
    OC = lucas_aligned[['OC_2018']].values
    BD = lucas_aligned[['pred_BD_median_LinReg_inf_BD_0-20_2018']].values # use predicted
    mineral_type = lucas_aligned[['HALA_2018gps_topsoil']].values
    all_Hassink += list(Hassink(Clay, Silt, OC, BD))
    all_Georgiou += list(Georgiou(Clay, Silt, mineral_type, BD))
    style = versions[version]
    c = selected_qmax_colors[version]
    axes[0].scatter(all_predicted_Qmax, all_Hassink, label=version, color=c, s=1, alpha=0.01)
    axes[1].scatter(all_predicted_Qmax, all_Georgiou, label=version, color=c, s=1, alpha=0.01)

    min = np.min([np.nanmin(all_predicted_Qmax), np.nanmin(all_Hassink), np.nanmin(all_Georgiou)])
    max = np.max([np.nanmax(all_predicted_Qmax), np.nanmax(all_Hassink), np.nanmax(all_Georgiou)])
    axes[0].plot([min, max], [min, max], 'k--', alpha=0.5)
    axes[1].plot([min, max], [min, max], 'k--', alpha=0.5)

for ax, name in zip(axes, ['Hassink', 'Georgiou']):
    ax.set_ylabel(rf"{name} Qmax / kg C m$^{{-3}}$")
    ax.set_xlabel(r"learned Qmax / mg C cm$^{-3}$")

legend_elements = []
for version in selected_Qmax_versions:
    style = versions[version]
    legend_elements.append(
        Line2D(
            [0], [0],
            color=selected_qmax_colors[version],
            linestyle=style["texture"],
            linewidth=style["linewidth"],
            label=style["legend"],
        )
    )
fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.,
    frameon=False)
plt.tight_layout()
plt.savefig("figures/Qmax.png", dpi=300, bbox_inches="tight")
# plt.show() # legend