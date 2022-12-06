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
    sys.path.append('./externals/nasa_subscriber/subscriber')


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

    
download_data_date = date.today() - timedelta(days=10)
download_raw_data(year = download_data_date.year, month = download_data_date.month, day = download_data_date.day)
