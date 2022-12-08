import cdsapi
import xarray as xr
import numpy as np
import os
import sys
sys.path.append('./externals/nasa_subscriber')
from datetime import date, timedelta, datetime

from subscriber import podaac_access as pa
from urllib.error import HTTPError
from urllib.request import urlretrieve
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def download_raw_data(year, month, day, raw_data_root='raw_data'):
    '''
    Download raw data using API

    * CyGNSS data
    * ERA5 data

    For compliance with the CyGNSSnet preprocessing routines, the data is stored in 

     > {raw_data_root}/{year}/{day-of-year}

    Parameters:
     year, month, day - download data from the full day specified
     raw_data_root    - root of path to store the data
    '''

    raw_data_sub = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y/%j")

    raw_data_dir = os.path.join(raw_data_root, raw_data_sub)

    print('Downloading data in this directory: ', raw_data_dir)

    start_date = datetime(year, month, day).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date   = datetime(year, month, day + 1).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f'--start-date {start_date}')
    print(f'--end-date   {end_date}')

    # PODAAC data
    adapted_podaac_downloader(start_date, end_date, raw_data_dir)

    # ERA5 data
    era5_downloader(year, month, day, raw_data_dir)


def era5_downloader(year, month, day, raw_data_dir):
    '''
    ERA5 data downloader from Copernicus

    We need to download all the time steps of the current day, as well as the 
    time step midnight on the following day. These are merged.

    Parameters:
      year, month, day - download data from the full day specified
      data_path  - path to store the data
    '''

    print("Start ERA5 download")
    target_data = os.path.join(raw_data_dir, 'ERA5_windspeed.nc')
    era5_data = os.path.join(raw_data_dir, 'ERA5_today.nc') 
    tomorrow_era5_data = os.path.join(raw_data_dir, 'ERA5_tomorrow.nc') 
    cds = cdsapi.Client() 
   
    # Retrieve today's data
    cds.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind',
            'total_precipitation',
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
            50, -180, -50, 180,
        ],
    },
    era5_data)

    # Retrieve tomorrow's data
    tomorrow = datetime(year, month, day) + timedelta(1)

    cds.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind',
            'total_precipitation',
        ],
        'year': tomorrow.year,
        'month': tomorrow.month,
        'day': tomorrow.day,
        'time': [
            '00:00', '01:00'
        ],
        'area': [
            50, -180, -50, 180,
        ],
    },
    tomorrow_era5_data)

    # Retrieve tomorrow's data
    with xr.open_dataset(era5_data) as f1, xr.open_dataset(tomorrow_era5_data) as f2:
        era5_ds = xr.merge([f1.load(), f2.load()])
        era5_ds.to_netcdf(target_data)

    print('SUCCESS: Retrieved ERA5 data')
    

def adapted_podaac_downloader(start_date, end_date, data_path):
    '''
    PODAAC data downloader adapted for CyGNSSnet

    Adapted from the run routine in
    https://github.com/podaac/data-subscriber/blob/main/subscriber/podaac_data_downloader.py

    Parameters:
     start_date - download start date in ISO format
     end_date   - download end date in ISO format
     data_path  - path to store the data
    '''

    # Default values
    page_size = 2000
    edl = pa.edl
    cmr = pa.cmr
    token_url = pa.token_url

    pa.setup_earthdata_login_auth(edl)
    token = pa.get_token(token_url, 'podaac-subscriber', edl)
    print('Completed PODAAC authentification')

    provider = 'POCLOUD'
    #search_cycles = args.search_cycles [None ?]
    short_name = 'CYGNSS_L1_V3.1'
    extensions = None
    #process_cmd = args.process_cmd [empty ?]

    download_limit = None
    ts_shift = timedelta(hours=0)

    verbose = True
    force = False


    if not os.path.isdir(data_path):
        print("NOTE: Making new data directory at " + data_path + "(This is the first run.)")
        os.makedirs(data_path, exist_ok=True)

    temporal_range = pa.get_temporal_range(start_date, end_date,
                                               datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))  # noqa E501
    params = [
        ('page_size', page_size),
        ('sort_key', "-start_date"),
        ('provider', provider),
        ('ShortName', short_name),
        ('temporal', temporal_range),
    ]
    print("Temporal Range: " + temporal_range)

    # TODO bbox

    #if args.bbox is not None:
    #    params.append(('bounding_box', args.bbox))

    # If 401 is raised, refresh token and try one more time
    try:
        results = pa.get_search_results(params, verbose)
    except HTTPError as e:
        if e.code == 401:
            token = pa.refresh_token(token, 'podaac-subscriber')
            params['token'] = token
            results = pa.get_search_results(params, verbose)
        else:
            raise e

    if verbose:
        print(str(results['hits']) + " granules found for " + short_name)  # noqa E501

    downloads_all = []
    downloads_data = [[u['URL'] for u in r['umm']['RelatedUrls'] if
                       u['Type'] == "GET DATA" and ('Subtype' not in u or u['Subtype'] != "OPENDAP DATA")] for r in
                      results['items']]
    downloads_metadata = [[u['URL'] for u in r['umm']['RelatedUrls'] if u['Type'] == "EXTENDED METADATA"] for r in
                          results['items']]
    checksums = pa.extract_checksums(results)

    for f in downloads_data:
        downloads_all.append(f)
    for f in downloads_metadata:
        downloads_all.append(f)

    downloads = [item for sublist in downloads_all for item in sublist]

    if len(downloads) >= page_size:
        logging.warning("Only the most recent " + str(
            page_size) + " granules will be downloaded; try adjusting your search criteria (suggestion: reduce time period or spatial region of search) to ensure you retrieve all granules.")

    # filter list based on extension
    if not extensions:
        extensions = pa.extensions
    filtered_downloads = []
    for f in downloads:
        for extension in extensions:
            if f.lower().endswith(extension):
                filtered_downloads.append(f)

    downloads = filtered_downloads

    print("Found " + str(len(downloads)) + " total files to download")
    if verbose:
        print("Downloading files with extensions: " + str(extensions))

    # NEED TO REFACTOR THIS, A LOT OF STUFF in here
    # Finish by downloading the files to the data directory in a loop.
    # Overwrite `.update` with a new timestamp on success.
    success_cnt = failure_cnt = skip_cnt = 0
    for f in downloads:
        try:
            output_path = os.path.join(data_path, os.path.basename(f))

            # decide if we should actually download this file (e.g. we may already have the latest version)
            if(os.path.exists(output_path) and not force and pa.checksum_does_match(output_path, checksums)):
                print(str(datetime.now()) + " SKIPPED: " + f)
                skip_cnt += 1
                continue

            urlretrieve(f, output_path)
            #pa.process_file(process_cmd, output_path, args)
            print(str(datetime.now()) + " SUCCESS: " + f)
            success_cnt = success_cnt + 1

            #if limit is set and we're at or over it, stop downloading
            if download_limit and success_cnt >= download_limit:
                break

        except Exception:
            logging.warning(str(datetime.now()) + " FAILURE: " + f, exc_info=True)
            failure_cnt = failure_cnt + 1

    print("Downloaded Files: " + str(success_cnt))
    print("Failed Files:     " + str(failure_cnt))
    print("Skipped Files:    " + str(skip_cnt))
    pa.delete_token(token_url, token)
    print("END\n\n")

if __name__=='__main__':    
    download_data_date = date.today() - timedelta(days=10)
    download_raw_data(year = download_data_date.year, month = download_data_date.month, day = download_data_date.day)
