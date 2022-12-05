#!/usr/bin/env python
# coding: utf-8

# # Preprocessing CyGNSS data

import os
import sys
import datetime
from datetime import date
from datetime import timedelta
sys.path.append('./cygnss-deployment/externals/gfz_cygnss/')
import argparse 
from gfz_202003.preprocessing import preprocess as prep
import numpy as np
import h5py
from matplotlib import pyplot as plt
import seaborn as sns
import xarray as xr
import cdsapi
from importlib import reload

def pre_processing():

    raw_data_root = '/home/harsh/Downloads/DKRZ/MLOps/2022-cygnss-deployment/raw_data'
    dev_data_dir = '/home/harsh/Downloads/DKRZ/MLOps/2022-cygnss-deployment/dev_data'     
        
    now = datetime.datetime.now()
    date = datetime.datetime(now.year, now.month, now.day) - timedelta(days=13)
    # date = datetime.datetime(2022, 9, 10)
    year  = date.year
    month = date.month
    day   = date.day


    raw_data_sub = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y/%j")

    raw_data_dir = os.path.join(raw_data_root, raw_data_sub)

    print(raw_data_dir)



    start_date = datetime.datetime(year, month, day).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date   = datetime.datetime(year, month, day + 1).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f'--start-date {start_date}')
    print(f'--end-date   {end_date}')

    era5_data = os.path.join(raw_data_dir, 'ERA5_windspeed.nc') 
    cds = cdsapi.Client() 
   
    cds.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind',
        ],
        'year': year,
        'month': month,
        'day': day,
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
    era5_data)

    era5_ds = xr.open_dataset(era5_data)


    for cygnss_file in os.listdir(raw_data_dir):
        if cygnss_file.startswith('cyg') and cygnss_file.endswith('.nc'):
            print(cygnss_file)
            annotate_dataset(os.path.join(raw_data_dir, cygnss_file), era5_data, save_dataset=True)       

    dday = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%j") # need that later
    
    
    reload(prep)

    args = argparse.Namespace(raw_data_dir='/work/ka1176/shared_data/2022-cygnss-deployment/raw_data/',
                        output_dir=dev_data_dir,
                        v_map=['brcs'],
                        n_valid_days=0,
                        n_test_days=1,
                        n_processes=1,
                        only_merge=False,
                        use_land_data=False,
                        is_ml_ops=True,
                        version='v3',
                        day=dday,
                        year=year,
                        reduce_mode='')

    prep.generate_input_data(args)

def annotate_dataset(cygnss_file, era5_file, save_dataset=False):
    '''
    Annotate a given CyGNSS dataset with ERA5 windspeed labels and save to disk
    
    Parameters:
    cygnss_file : path to CyGNSS dataset
    era5_file   : path to orresponding ERA5 dataset
    save_dataset : if True, save dataset to disk overwriting cygnss_file (default: False)
    
    Returns:
    Annotated CyGNSS dataset
    '''
    
    # necessary because lazy loading prohibits overwriting the netcdf files at the end of this section
    with xr.open_dataset(cygnss_file) as data:
        cygnss_ds = data.load()
        
    with xr.open_dataset(era5_file) as data:
        era5_ds = data.load()
        
    # needs to be shifted by 180 for compatibility with CyGNSS
    era5_ds = era5_ds.assign_coords(longitude=era5_ds.coords['longitude'] + 180)
    
    interp_ds = era5_ds.interp(longitude=cygnss_ds.sp_lon, latitude=cygnss_ds.sp_lat, time=cygnss_ds.ddm_timestamp_utc)
    
    cygnss_ds['ERA5_u10'] = interp_ds['u10']
    cygnss_ds['ERA5_v10'] = interp_ds['v10']

    tmp_attrs = cygnss_ds['ERA5_u10'].attrs
    tmp_attrs['long_name'] = cygnss_ds['ERA5_u10'].long_name + ' (interpolated)'
    cygnss_ds['ERA5_u10'].attrs = tmp_attrs

    tmp_attrs = cygnss_ds['ERA5_v10'].attrs
    tmp_attrs['long_name'] = cygnss_ds['ERA5_v10'].long_name + ' (interpolated)'
    cygnss_ds['ERA5_v10'].attrs = tmp_attrs
    
    cygnss_ds = cygnss_ds.drop_vars(['longitude', 'latitude', 'time'])
    
    # dummy values only for preprocessing routine
    cygnss_ds['GPM_precipitation'] = -9999
    cygnss_ds['ERA5_mdts'] = -9999
    cygnss_ds['ERA5_mdww'] = -9999
    cygnss_ds['ERA5_swh'] = -9999
    cygnss_ds['ERA5_shts'] = -9999
    cygnss_ds['ERA5_shww'] = -9999
    cygnss_ds['ERA5_p140121'] = -9999
    cygnss_ds['ERA5_p140124'] = -9999
    cygnss_ds['ERA5_p140127'] = -9999
    
    if save_dataset:
        cygnss_ds.to_netcdf(cygnss_file)
        
    return cygnss_ds



# pre_processing()