## 1) Files to have locally
- `original_files/lucas_harmo_uf.csv`
- `original_files/LUCAS-SOIL-2018.csv`
- `original_files/LUCAS_Topsoil_2015_20200323.csv`
- `original_files/Romania.csv`
- `original_files/Bulgaria.csv`
- `original_files/LUCAS_TOPSOIL_v1.xlsx`
- `original_files/LCS.csv`
- `original_files/microbialProperties_SOC_data.csv`
- `original_files/BulkDensity_2018_final-2.csv`
- `original_files/Boundary_df_th_coords.rds`
- `original_files/BNPP/BNPP_0-20cm.tif`
- `original_files/BNPP/BNPP_20-40cm.tif`
- `original_files/BNPP/BNPP_40-60cm.tif`
- `original_files/BNPP/BNPP_60-80cm.tif`
- `original_files/BNPP/BNPP_80-100cm.tif`
- `original_files/BNPP/BNPP_100-150cm.tif`
- `original_files/BNPP/BNPP_150-200cm.tif`
- `original_files/BNPP/BNPP_0-200cm.tif`

## 2) Set up environment
```bash
micromamba create -y -n hybrid-lucas -f environment-hybrid-lucas.yml
```

## 3) Run
```bash
micromamba run -n hybrid-lucas papermill 1_harmonize_all_LUCAS.ipynb 1_harmonize_all_LUCAS.ipynb
micromamba run -n hybrid-lucas papermill 2_preprocess.ipynb 2_preprocess.ipynb
micromamba run -n hybrid-lucas papermill 3_add_gee.ipynb 3_add_gee.ipynb
micromamba run -n hybrid-lucas papermill 4_add_nc.ipynb 4_add_nc.ipynb
micromamba run -n hybrid-lucas python 5_predict.py
micromamba run -n hybrid-lucas python 6_hybrid.py
```