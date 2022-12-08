import sys
from API import download_raw_data
from datetime import datetime, timedelta, date
from Preprocessing import pre_processing

def download_data(year, month, day, raw_data_root):
    # Using API calls
    download_raw_data(year, month, day, raw_data_root=raw_data_root)
    
def main(offset):

    # Define the date and pass it to the individual tasks
    download_date = date.today() - timedelta(days=int(offset))
    date_ = download_date.strftime("%Y-%m-%d")

    raw_data_root = '/work/ka1176/shared_data/2020-03/raw_data_v3-1'

    print("*"*50)
    print("  Download date", date_)
    print("*"*50)

    # Download data for the past 10th day from today, today - 10th day
    download_data(download_date.year, download_date.month, download_date.day, raw_data_root)

    # annotate data
    # create filtered hdf5 from preprocessing
    pre_processing(download_date.year, download_date.month, download_date.day, '/scratch/k202141/')

if __name__ == "__main__":    

    main(sys.argv[1])
