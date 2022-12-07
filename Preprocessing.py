#!/usr/bin/env python
# coding: utf-8

# # Preprocessing CyGNSS data

import os
import sys
from datetime import datetime, date, timedelta
import argparse 

sys.path.append('externals/gfz_cygnss/')
from gfz_202003.preprocessing import preprocess as prep
#sys.path.append('externals/gfz_cygnss/gfz_202003')
#from preprocessing import preprocess as prep

import numpy as np
import xarray as xr
import hashlib

def pre_processing(year, month, day, dev_data_dir='./dev_data'):
    '''
    Preprocessing routines for CyGNSSnet

    (1) Annotate CyGNSS raw data with windspeed labels from ERA5
    (2) Filter and generate hdf5 file

    Folder structure:

    * raw_data
    * annotated_raw_data
    * dev_data : filtered, one file test_data.h5

    Parameters:
      year, month, day - preprocess the data downloaded for that day
      dev_data_dir     - directory to store the filtered data for that day

    Returns:
      h5_file - path to the filtered data for that day
    '''

    raw_data_root = './raw_data'
    annotated_raw_data_root = './annotated_raw_data'
        
    raw_data_sub = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y/%j")

    raw_data_dir = os.path.join(raw_data_root, raw_data_sub)
    annotated_raw_data_dir = os.path.join(annotated_raw_data_root, raw_data_sub)
    era5_data = os.path.join(raw_data_dir, 'ERA5_windspeed.nc') 

    if not os.path.isdir(annotated_raw_data_dir):
        os.makedirs(annotated_raw_data_dir, exist_ok=True)

    if not os.path.isdir(dev_data_dir):
        os.makedirs(dev_data_dir, exist_ok=True)

    start_date = datetime(year, month, day).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date   = datetime(year, month, day + 1).strftime("%Y-%m-%dT%H:%M:%SZ")

    for cygnss_file in os.listdir(raw_data_dir):
        if cygnss_file.startswith('cyg') and cygnss_file.endswith('.nc'):
            print("annotating", cygnss_file)

            pcf = os.path.join(raw_data_dir, cygnss_file)
            phf = os.path.join(raw_data_dir, cygnss_file.replace('.nc', '.md5'))

            print("create hash", phf)

            if os.path.exists(phf):
                print("-- hash exists, skip")
                continue 

            annotate_dataset(pcf, era5_data, save_dataset=True)       

            hmd5 = hash_large_file(pcf)
            with open(phf, 'w') as hf:
                hf.write(hmd5)

    dday = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%j") # need that later
    
    args = argparse.Namespace(raw_data_dir=annotated_raw_data_root,
                        output_dir=dev_data_dir,
                        v_map=['brcs', 'eff_scatter', 'raw_counts', 'power_analog'],
                        n_valid_days=0,
                        n_test_days=1,
                        n_processes=1,
                        only_merge=False,
                        use_land_data=False,
                        is_ml_ops=True,
                        version='v3.1',
                        day=dday,
                        year=year,
                        reduce_mode='')

    prep.generate_input_data(args)

def hash_large_file(file):
    '''
    Read a large file in chunks and compute the MD5 checksum

    Parameters:
     file - the file to be hashed

    Returns:
     hash(file)
    '''
    with open(file,'rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    print(file_hash.hexdigest())
    return file_hash.hexdigest()

def annotate_dataset(cygnss_file, era5_file, save_dataset=False):
    '''
    Annotate a given CyGNSS dataset with ERA5 windspeed labels and save to disk

    Annotate additional ERA5 parameters (GPM_precipitation)

    TODO: hash
    
    Parameters:
    cygnss_file : path to CyGNSS dataset
    era5_file   : path to orresponding ERA5 dataset
    save_dataset : if True, save dataset to disk in annotated_raw_data_dir (default: False)
    
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
    cygnss_ds['GPM_precipitation'] = interp_ds['tp']

    tmp_attrs = cygnss_ds['ERA5_u10'].attrs
    tmp_attrs['long_name'] = cygnss_ds['ERA5_u10'].long_name + ' (interpolated)'
    cygnss_ds['ERA5_u10'].attrs = tmp_attrs

    tmp_attrs = cygnss_ds['ERA5_v10'].attrs
    tmp_attrs['long_name'] = cygnss_ds['ERA5_v10'].long_name + ' (interpolated)'
    cygnss_ds['ERA5_v10'].attrs = tmp_attrs
    
    cygnss_ds = cygnss_ds.drop_vars(['longitude', 'latitude', 'time'])
    
    # dummy values only for preprocessing routine
    cygnss_ds['ERA5_mdts'] = -9999
    cygnss_ds['ERA5_mdww'] = -9999
    cygnss_ds['ERA5_swh'] = -9999
    cygnss_ds['ERA5_shts'] = -9999
    cygnss_ds['ERA5_shww'] = -9999
    cygnss_ds['ERA5_p140121'] = -9999
    cygnss_ds['ERA5_p140124'] = -9999
    cygnss_ds['ERA5_p140127'] = -9999
    
    if save_dataset:
        cygnss_ds.to_netcdf(cygnss_file.replace('raw_data', 'annotated_raw_data'))
        
    return cygnss_ds

if __name__=='__main__':
    pre_processing()
