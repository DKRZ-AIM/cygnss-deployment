import os
from pydoc import cli
import sys
import time
import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary
from sklearn.metrics import mean_squared_error
from collections import namedtuple
import xarray
import mlflow
from prefect import flow, task
import streamlit as st
# TODO Fix these imports
# from prefect.deployments import DeploymentSpec
#from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.deployments import Deployment
from prefect.filesystems import RemoteFileSystem
from prefect.infrastructure import DockerContainer
from prefect.task_runners import SequentialTaskRunner
from pymongo import MongoClient, errors
from API import download_raw_data
from datetime import datetime, timedelta
sys.path.append('./2020-03-gfz-remote-sensing')
sys.path.append('./2020-03-gfz-remote-sensing/gfz_202003')
sys.path.append('./2020-03-gfz-remote-sensing/gfz_202003/training')

from cygnssnet import ImageNet, DenseNet, CyGNSSNet, CyGNSSDataModule, CyGNSSDataset
import plots

@task
def download_data():
    download_data_date = datetime.date.today() - datetime.timedelta(days=10)
    download_raw_data(year = download_data_date.year, month = download_data_date.month, day = download_data_date.day)
    
@task
def get_data(client):        
        cygnss = client.cygnss                        
        items = cygnss.cygnss_collection.find()        
        items = list(items)  # make hashable for st.experimental_memo
        for item in items:
            print(f"RMSE is: {item['rmse']}")            
        
        
@task
def drop_database(client):
    client.drop_database('cygnss')

@task
@st.experimental_singleton
def save_to_db(domain, port, y_pred, rmse, date, rmse_time):
    # use a try-except indentation to catch MongoClient() errors
    try:
        print('entering mongo db connection')
        
     
        
        client = MongoClient(
        host = [ str(domain) + ":" + str(port) ],
        serverSelectionTimeoutMS = 3000, # 3 second timeout
        username = "root",
        password = "example",
    )

       # uncomment and if you wanna clear out the data
        client.drop_database('cygnss')

        # print the version of MongoDB server if connection successful
        print ("server version:", client.server_info()["version"])
        
        data = {
                "rmse": rmse.tolist(),
                "bin_rmse": rmse_time["rmse"].tolist(),
                "bin_bias": rmse_time["bias"].tolist(),
                "bin_counts": rmse_time["counts"].tolist(),
                "event_date": date,
                "scatterplot_path": f"{os.path.dirname(__file__)}/plots/scatter.png",
                "histogram_path": f"{os.path.dirname(__file__)}/plots/histo.png",
                "era_average_path": f"{os.path.dirname(__file__)}/plots/era_average.png",
                "rmse_average_path": f"{os.path.dirname(__file__)}/plots/rmse_average.png",
                "today_longrunavg_path": f"{os.path.dirname(__file__)}/plots/today_longrunavg.png",
                "today_long_bias_path": f"{os.path.dirname(__file__)}/plots/today_long_bias.png",
                "sample_counts_path": f"{os.path.dirname(__file__)}/plots/sample_counts.png",
                "rmse_bins_era_path": f"{os.path.dirname(__file__)}/plots/rmse_bins_era.png",
                "bias_bins_era_path": f"{os.path.dirname(__file__)}/plots/bias_bins_era.png",
                "y_pred": y_pred.tolist()
                }

        cygnss_collection = client["cygnss"].cygnss_collection


        cygnss_collection = cygnss_collection.insert_many([data])

        print(f"Multiple tutorials: {cygnss_collection.inserted_ids}")

    except errors.ServerSelectionTimeoutError as err:
        # set the client and DB name list to 'None' and `[]` if exception
        client = None
        # catch pymongo.errors.ServerSelectionTimeoutError
        print (err)
    

@task
def get_hyper_params(model_path, model, data_path):
    # Note for future: for fixed model write h_params in config file
    checkpoint = torch.load(os.path.join(model_path, model),
                    map_location=torch.device("cpu"))
    checkpoint['hyper_parameters']["data"] = data_path
    col_idx_lat = checkpoint["hyper_parameters"]["v_par_eval"].index('sp_lat')
    col_idx_lon = checkpoint["hyper_parameters"]["v_par_eval"].index('sp_lon')
    args = namedtuple("ObjectName", checkpoint['hyper_parameters'].keys())\
            (*checkpoint['hyper_parameters'].values())
    return args, col_idx_lat, col_idx_lon 

@task
def get_backbone(args, input_shapes):
    if args.model=='cnn':
        backbone = ImageNet(args, input_shapes)
    elif args.model=='dense':
        backbone = DenseNet(args, input_shapes)
    return backbone

@task 
def make_predictions(test_loader, model):
    trainer = pl.Trainer(enable_progress_bar=False)
    trainer.test(model=model, dataloaders=test_loader)
    y_pred = trainer.predict(model=model, dataloaders=[test_loader])
    y_pred = torch.cat(y_pred).detach().cpu().numpy().squeeze()
    return y_pred

@task
def rmse_bins(y_true, y_pred, y_bins):
    # Find the indices for the windspeed bins - below 12 m/s, below 16 m/s, above 16 m/s
    y_ix   = np.digitize(y_true, y_bins, right=False)

    all_rmse = np.zeros(len(y_bins))
    all_bias = np.zeros(len(y_bins))
    all_counts = np.zeros(len(y_bins))

    for i, yy in enumerate(y_bins):
        if np.any(y_ix==i):
            rmse = mean_squared_error(y_true[y_ix==i], y_pred[y_ix==i], squared=False)
            all_rmse[i] = rmse
            all_bias[i] = np.mean(y_pred[y_ix==i] - y_true[y_ix==i])
            all_counts[i] = np.sum(y_ix==i)
        else:
            all_rmse[i] = None
            all_bias[i] = None
            all_counts[i] = 0
        df_rmse = pd.DataFrame(dict(rmse=all_rmse, bias=all_bias, bins=y_bins, counts=all_counts))
        return df_rmse

@task
def rmse_over_time(y_bins, df_rmse):
    # mock up data that represents the long running average rmse
    df_rmse["time"] = "today"
    
    df_mockup = pd.DataFrame(dict(bins=y_bins, 
                   rmse=df_rmse["rmse"] + np.random.rand(len(y_bins))-0.5, 
                   bias=df_rmse["bias"] + np.random.rand(len(y_bins))-0.5,
                   counts=df_rmse["counts"] * 1000))
    df_mockup["time"] = "long-running average"

    df_mockup = pd.concat([df_rmse, df_mockup], ignore_index=True)
    return df_mockup

@flow
def main():

    # Download data for the past 10th day from today, today - 10th day
    #download_data()
    
    # TODO
    # pre_process()

    # TODO: get date from preprocessing
    now = datetime.now()
    date = datetime(now.year, now.month, now.day) - timedelta(days=10)
    #date = datetime.datetime(2022, 9, 10)
    
    model_path = './2022-cygnss-deployment/'\
            'cygnss_trained_model/ygambdos_yykDM/checkpoint'
    model = 'cygnssnet-epoch=0.ckpt'
    data_path = './2022-cygnss-deployment/small_data/' #'../data' # TODO, change the path outside of code, in a separete folder
    h5_file = h5py.File(os.path.join(data_path, 'test_data.h5'), 'r', rdcc_nbytes=0)

    mlflow.set_tracking_uri("sqlite:///mlruns.db") # TODO: change this to other db
    mlflow.set_experiment("cygnss")

 
    # get hyperparameters 
    args, col_idx_lat, col_idx_lon  = get_hyper_params.submit(model_path, model, data_path).result()

    cdm = CyGNSSDataModule(args)
    cdm.setup(stage='test')
    input_shapes = cdm.get_input_shapes(stage='test')
    backbone = get_backbone.submit(args, input_shapes).result()
  
    # load model
    cygnss_model = CyGNSSNet.load_from_checkpoint(os.path.join(model_path, model),
                                           map_location=torch.device('cpu'), 
                                           args=args, 
                                           backbone=backbone)
    cygnss_model.eval()

    test_loader = cdm.test_dataloader()    
    # make predictions
    y_pred = make_predictions(test_loader, cygnss_model)
    
    # get true labels
    dataset = CyGNSSDataset('test', args)
    y = dataset.y

    # calculate rmse
    y_bins = [4, 8, 12, 16, 20, 100]
    df_rmse = rmse_bins.submit(y, y_pred, y_bins).result()
    df_mockup = rmse_over_time.submit(y_bins, df_rmse).result()
    with mlflow.start_run():
        rmse = mean_squared_error(y, y_pred, squared=False)
        mlflow.log_metric('rmse', rmse)
   
    # make plots
    sp_lat = test_loader.dataset.v_par_eval[:, col_idx_lat]
    sp_lon = test_loader.dataset.v_par_eval[:, col_idx_lon]
    plots.make_scatterplot(y, y_pred)
    plots.make_histogram(y, y_pred)
    # plots.era_average(y, sp_lon, sp_lat)
    # plots.rmse_average(y, y_pred, sp_lon, sp_lat)
    plots.today_longrunavg(df_mockup, y_bins)
    plots.today_longrunavg_bias(df_mockup, y_bins)
    plots.sample_counts(df_rmse, y_bins)
    plots.rmse_bins_era(df_rmse, y_bins)
    plots.bias_bins_era(df_rmse, y_bins)
    # global variables for MongoDB host (default port is 27017)
    DOMAIN = 'mongodb'
    PORT = 27017
    
    # Save results to the mongo database
    save_to_db(domain=DOMAIN, port=PORT, y_pred=y_pred, \
            rmse=rmse, date=date, rmse_time=df_rmse)


if __name__ == "__main__":    

    deployment = Deployment.build_from_flow(
        schedule = IntervalSchedule(interval=timedelta(minutes=2)),
        flow=main,  
        name="cygnss",  
        work_queue_name="demo"
)
    deployment.apply()

