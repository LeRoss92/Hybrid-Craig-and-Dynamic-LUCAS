# attach to dataset
dataset_path = "3_with_gee.pkl"

# extract from:
links = [
    {
        'name': "HALA",
        'subs': ['topsoil'],
        'links': ['original_files/topsoil_major_activity_type.nc']
    },
    {
        'name': "BNPP",
        'subs': ['0-20', '20-40', '40-60', '60-80', '80-100', '100-150', '150-200', '0-200'],
        'links': ['/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_0-20cm.tif',
                  '/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_20-40cm.tif',
                  '/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_40-60cm.tif',
                  '/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_60-80cm.tif',
                  '/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_80-100cm.tif',
                  '/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_100-150cm.tif',
                  '/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_150-200cm.tif',
                  '/User/homes/leorossd/people/HPLG/examples/021_data_4/original_files/BNPP/BNPP_0-200cm.tif'
                  ]
    },
    {
        'name': "MODIS_NPP",
        'subs': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
        'links': ['/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2001.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2002.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2003.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2004.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2005.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2006.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2007.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2008.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2009.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2010.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2011.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2012.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2013.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2014.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2015.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2016.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2017.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2018.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2019.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2020.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2021.nc',
                    '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD17A3HGF.061/Data/Npp/Npp.org.86400.43200.2022.nc']
    },
    {
        'name': "BIOCLIM",
        'subs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'links': ["/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_1.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_2.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_3.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_4.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_5.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_6.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_7.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_8.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_9.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_10.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_11.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_12.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_13.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_14.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_15.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_16.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_17.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_18.nc",
                "/Net/Groups/BGI/scratch/leorossd/GridRefDB/data/gridded_products/nc_100x100_chunked/WorldClim_historical_climate/bioclimatic_30s/wc2.1_30s_bio_19.nc"
                ]
    }
]

# save to:
output_path = "4_with_nc.pkl"

import netCDF4 as nc
import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import transform
import numpy as np
def inquire_netcdf_values(nc_file_path, coordinates):
    if nc_file_path.endswith('.tif') or nc_file_path.endswith('.tiff'):
        return _extract_from_tif(nc_file_path, coordinates)
    return _extract_from_netcdf(nc_file_path, coordinates)
def _extract_from_tif(tif_file_path, coordinates):
    with rasterio.open(tif_file_path) as src:
        src_crs = src.crs
        raster_coords = []
        for lat, lon in coordinates:
            if src_crs != 'EPSG:4326':
                x, y = transform('EPSG:4326', src_crs, [lon], [lat])
                raster_coords.append((x[0], y[0]))
            else:
                raster_coords.append((lon, lat))
        values = []
        for x, y in raster_coords:
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                values.append(float(value))
            else:
                values.append(np.nan)
        return values
def _extract_from_netcdf(nc_file_path, coordinates):
    dataset_nc = nc.Dataset(nc_file_path)
    try:
        lat_str = 'latitude'
        lon_str = 'longitude'
        lat = dataset_nc.variables[lat_str][:]
        lon = dataset_nc.variables[lon_str][:]
    except:
        lat_str = 'lat'
        lon_str = 'lon'
        lat = dataset_nc.variables[lat_str][:]
        lon = dataset_nc.variables[lon_str][:]
    band_names = ''
    any_band = []
    for var_name in dataset_nc.variables:
        if var_name not in ['crs', 'time', lat_str, lon_str]:
            band_names += var_name + ', '
            any_band.append(var_name)
    band_names = [band_names[:-2]]
    var = dataset_nc.variables[any_band[0]]
    try:
        chunking = var.chunking()
        if chunking != 'contiguous' and isinstance(chunking, (tuple, list)) and len(chunking) >= 2:
            lat_chunk_size, lon_chunk_size = chunking[0], chunking[1]
        else:
            lat_chunk_size = lon_chunk_size = None
    except Exception:
        lat_chunk_size = lon_chunk_size = None
    ds = xr.open_dataset(nc_file_path)
    if lat_chunk_size is not None and lon_chunk_size is not None:
        ds = ds.chunk({lat_str: lat_chunk_size, lon_str: lon_chunk_size})
    latitudes = np.array([coord[0] for coord in coordinates])
    longitudes = np.array([coord[1] for coord in coordinates])
    values_at_coords = ds[any_band[0]].sel({lat_str: xr.DataArray(latitudes, dims='points'), lon_str: xr.DataArray(longitudes, dims='points')}, method='nearest')
    npp_values = values_at_coords.compute()
    return [float(value) for value in npp_values]

import pickle

with open(dataset_path, 'rb') as f:
    df = pickle.load(f)

def add(lat_str, long_str, gps_year):
    global df
    lat = df[lat_str].to_numpy(dtype=float)
    lon = df[long_str].to_numpy(dtype=float)

    valid_mask = np.isfinite(lat) & np.isfinite(lon)

    coordinates = list(zip(lat[valid_mask], lon[valid_mask]))

    new_cols = {}
    for new in links:
        for i, sub in enumerate(new['subs']):
            col = f"{new['name']}_{gps_year}gps_{sub}"
            print(col)

            values = np.full(len(df), np.nan)

            if coordinates:
                queried = inquire_netcdf_values(new['links'][i], coordinates)
                values[valid_mask] = queried

            new_cols[col] = values

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

add('gps_lat_2009', 'gps_long_2009', 2009)
add('gps_lat_2015', 'gps_long_2015', 2015)
add('gps_lat_2018', 'gps_long_2018', 2018)


df.to_pickle("4_with_nc.pkl")