import os
from pydoc import cli
import sys
import time
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary
from sklearn.metrics import mean_squared_error
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import lines, colors, ticker
import seaborn as sns
import xarray
import mlflow
from prefect import flow, task
import streamlit as st
# TODO Fix these imports
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.task_runners import SequentialTaskRunner
from pymongo import MongoClient, errors
#from API import download_raw_data
from datetime import datetime, timedelta
sys.path.append('./2020-03-gfz-remote-sensing')
sys.path.append('./2020-03-gfz-remote-sensing/gfz_202003')
sys.path.append('./2020-03-gfz-remote-sensing/gfz_202003/training')

from cygnssnet import ImageNet, DenseNet, CyGNSSNet, CyGNSSDataModule, CyGNSSDataset


@task
def download_data():
    download_data_date = datetime.date.today() - datetime.timedelta(days=10)
    download_raw_data(year = download_data_date.year, month = download_data_date.month, day = download_data_date.day)
    
#@task
#def write_data(client):
#        data_1 = {
#        "rmse": 3.1, 
#        "event_date":  datetime.datetime(2022, 8, 10),
#        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
#        }

#        data_2 = {
#        "rmse": 2.1,         
#        "event_date":  datetime.datetime(2022, 8, 9),
#        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
#        }

#        data_3 = {
#        "rmse": 3.2,         
#        "event_date":  datetime.datetime(2022, 8, 8),
#        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
#        }


#        cygnss_collection = client["cygnss"].cygnss_collection

#        cygnss_collection = cygnss_collection.insert_many([data_1, data_2, data_3])

#        print(f"Multiple tutorials: {cygnss_collection.inserted_ids}")

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
def save_to_db(domain, port, y_pred, rmse, date, all_rmse):
    # use a try-except indentation to catch MongoClient() errors
    try:
        print('entering mongo db connection')
        
        # uncomment and if you wanna clear out the data
        # client.drop_database('cygnss')
        
        client = MongoClient(
        host = [ str(domain) + ":" + str(port) ],
        serverSelectionTimeoutMS = 3000, # 3 second timeout
        username = "root",
        password = "example",
    )

        # print the version of MongoDB server if connection successful
        print ("server version:", client.server_info()["version"])

        data_1 = {
        "rmse": 3.1, 
        "event_date":  datetime(2022, 8, 10),
        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
        }

        data_2 = {
        "rmse": 2.1,         
        "event_date":  datetime(2022, 8, 9),
        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
        }

        data_3 = {
        "rmse": 3.2,         
        "event_date":  datetime(2022, 8, 8),
        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
        }


        data_4 = {
                "rmse": rmse.tolist(),
                "all_rmse": all_rmse.tolist(),
                "event_date": date,
                "scatterplot_path": f"{os.path.dirname(__file__)}/plots/scatter.png",
                "histogram_path": f"{os.path.dirname(__file__)}/plots/histo.png",
                #"y_pred": pymongo.binary.Binary( pickle.dumps(y_pred, protocol=2)))
                }

        cygnss_collection = client["cygnss"].cygnss_collection

        cygnss_collection = cygnss_collection.insert_many([data_1, data_2, data_3, data_4])

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
    args = namedtuple("ObjectName", checkpoint['hyper_parameters'].keys())\
            (*checkpoint['hyper_parameters'].values())
    return args 

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
def rmse_bins(y_true, y_pred):
    # Find the indices for the windspeed bins - below 12 m/s, below 16 m/s, above 16 m/s
    y_bins = [4, 8, 12, 16, 20, 100]
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
        return all_rmse

@task
def make_scatterplot(y_true, y_pred):
    ymin = 2.5
    ymax = 25.0

    fig=plt.figure()
    ax=fig.add_subplot(111)

    img=ax.hexbin(y_true, y_pred, cmap='viridis', norm=colors.LogNorm(vmin=1, vmax=25000), mincnt=1)
    clb=plt.colorbar(img)
    clb.set_ticks([1, 10, 100, 1000, 10000])
    clb.set_ticklabels([r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$'])
    clb.set_label('Samples in bin')
    clb.ax.tick_params()

    ax.set_xlabel('ERA5 wind speed (m/s)')
    ax.set_ylabel('Predicted wind speed (m/s)')

    ax.plot(np.linspace(0, 30), np.linspace(0, 30), 'w:')

    ax.set_ylim(ymin, 25)
    ax.set_xlim(ymin, 25)

    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    ax.set_yticks([5, 10, 15, 20, 25])
    ax.set_yticklabels([5, 10, 15, 20, 25])

    fig.tight_layout()
    plt.savefig(f'{os.path.dirname(__file__)}/plots/scatter.png')

@task 
def make_histogram(y_true, y_pred):
    fig=plt.figure()
    ax=fig.add_subplot(111)

    sns.histplot(y_true, ax=ax, color='C7', label='ERA5 wind speed (m/s)')
    sns.histplot(y_pred, ax=ax, color='C2', label='Predicted wind speed (m/s)')

    ax.legend(fontsize=12)

    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticklabels([5, 10, 15, 20, 25])
    ax.set_xlabel('ERA5 wind speed (m/s)')
    plt.savefig(f'{os.path.dirname(__file__)}/plots/histo.png')

@flow(task_runner=SequentialTaskRunner())
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

 
    # get hyper parameters 
    args = get_hyper_params(model_path, model, data_path).result()

    cdm = CyGNSSDataModule(args)
    cdm.setup(stage='test')
    input_shapes = cdm.get_input_shapes(stage='test')
    backbone = get_backbone(args, input_shapes).result()
  
    # load model
    cygnss_model = CyGNSSNet.load_from_checkpoint(os.path.join(model_path, model),
                                           map_location=torch.device('cpu'), 
                                           args=args, 
                                           backbone=backbone)
    cygnss_model.eval()

    test_loader = cdm.test_dataloader()    
    # make predictions
    y_pred = make_predictions(test_loader, cygnss_model).result()
    
    # get true labels
    dataset = CyGNSSDataset('test', args)
    y = dataset.y

    # calculate rmse
    all_rmse = rmse_bins(y, y_pred)
    with mlflow.start_run():
        rmse = mean_squared_error(y, y_pred, squared=False)
        mlflow.log_metric('rmse', rmse)
   
    # make plots
    make_scatterplot(y, y_pred)
    make_histogram(y, y_pred)

    # global variables for MongoDB host (default port is 27017)
    DOMAIN = 'mongodb'
    PORT = 27017

    # Save results to the mongo database
    save_to_db(domain=DOMAIN, port=PORT, y_pred=y_pred, rmse=rmse, date=date, all_rmse=all_rmse)

main()

#DeploymentSpec(
#    flow=main,
#    name="model_inference",
#    schedule=IntervalSchedule(interval=timedelta(minutes=2)),
#    flow_runner=SubprocessFlowRunner(),
#    tags=["cygnss"]
#)

