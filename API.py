import subprocess
import cdsapi
import xarray as xr
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from global_land_mask import globe
import pandas as pd
import os
import sys
import datetime
from datetime import date
from datetime import timedelta

def download_raw_data(year  = 2021, month = 3, day   = 17):
    # # APIs
    # Retrieve CyGNSS and ERA5 data using APIs
    raw_data_root = './raw_data/'
    dev_data_root = './dev_data/'


    sys.path.append('./externals/gfz_cygnss/')
    sys.path.append('./externals/nasa_subscriber/')


    raw_data_sub = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y/%j")

    raw_data_dir = os.path.join(raw_data_root, raw_data_sub)

    print('Downloading data in this directory: ', raw_data_dir)

    start_date = datetime.datetime(year, month, day).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date   = datetime.datetime(year, month, day + 1).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f'--start-date {start_date}')
    print(f'--end-date   {end_date}')
    
    # Download the data
    os.system(f"python ./externals/nasa_subscriber/subscriber/podaac_data_downloader.py  -c CYGNSS_L1_V3.1 -d {raw_data_dir} --start-date {start_date} --end-date {end_date}")
   
    # if we want to use sub process in place of os.system above
    # subprocess.call(['python', './2022-cygnss-deployment/data-subscriber/subscriber/podaac_data_downloader.py', '-c ' + raw_data_dir, '--start-date ' + start_date, '--end-date ' + end_date])

    
    cds = cdsapi.Client()

    cds.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': '2022',
            'month': '08',
            'day': '02',
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00'
            ],
            'area': [
                40, -180, -40, 180,
            ],
        },
        './2022-cygnss-deployment/raw_data/download.nc')

    ds_era5 = xr.open_dataset('./2022-cygnss-deployment/raw_data/download.nc')
    ds_era5.close()




    ds_era5['ERA5_windspeed'] = np.sqrt( ds_era5['u10'] * ds_era5['u10'] + ds_era5['v10'] * ds_era5['v10'] )


    lats = ds_era5['latitude'].values
    lons = ds_era5['longitude'].values

    times = ds_era5['time'].values

    print('Total number of grid points:', len(lons) * len(lats))



    mesh_lats, mesh_lons = np.meshgrid(lats, lons)

    print(len(mesh_lats.flatten()))



    globe_ocean_mask = globe.is_ocean(mesh_lats, mesh_lons)
    globe_land_mask = ~globe_ocean_mask

    print('Fraction of ocean grid points:', globe_ocean_mask.sum() / len(globe_ocean_mask.flatten()))


download_data_date = date.today() - timedelta(days=13)
download_raw_data(year = download_data_date.year, month = download_data_date.month, day = download_data_date.day)
