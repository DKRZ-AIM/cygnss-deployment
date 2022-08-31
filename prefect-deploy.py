import os
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
import xarray
import mlflow
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.task_runners import SequentialTaskRunner
from pymongo import MongoClient, errors

sys.path.append('../../caroline/gitlab/2020-03-gfz-remote-sensing/')
from cygnssnet import ImageNet, DenseNet, CyGNSSNet, CyGNSSDataModule, CyGNSSDataset
import gfz_202003.utils.mathematics as mat

@task
def mongo_db_connection(domain, port):
    # use a try-except indentation to catch MongoClient() errors
    try:
        # try to instantiate a client instance
        client = MongoClient(
            host = [ str(domain) + ":" + str(port) ],
            serverSelectionTimeoutMS = 3000, # 3 second timeout
            #username = "root",
            #password = "1234",
        )

        # print the version of MongoDB server if connection successful
        print ("server version:", client.server_info()["version"])

        # get the database_names from the MongoClient()
        database_names = client.list_database_names()

    except errors.ServerSelectionTimeoutError as err:
        # set the client and DB name list to 'None' and `[]` if exception
        client = None
        database_names = []

        # catch pymongo.errors.ServerSelectionTimeoutError
        print ("pymongo ERROR:", err)

    return client

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
def save_to_db():
    pass

@flow(task_runner=SequentialTaskRunner())
def main():

    model_path = '/work/ka1176/shared_data/2022-cygnss-deployment/'\
            'cygnss_trained_model/ygambdos_yykDM/checkpoint'
    model = 'cygnssnet-epoch=0.ckpt'
    data_path = '../../shared_data/2022-cygnss-deployment/small_data/' #'../data'
    h5_file = h5py.File(os.path.join(data_path, 'test_data.h5'), 'r', rdcc_nbytes=0)

    mlflow.set_tracking_uri("sqlite:///mlflow.db") # future todo: change this to other db
    mlflow.set_experiment("cygnss")

    # global variables for MongoDB host (default port is 27017)
    DOMAIN = 'localhost'
    PORT = 27017
    #client = mongo_db_connection(domain=DOMAIN, port=PORT)

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
    with mlflow.start_run():
        rmse = mean_squared_error(y, y_pred, squared=False)
        mlflow.log_metric('rmse', rmse)

    
    # todo save results in db
    # save_to_tb()

main()

# DeploymentSpec(
#    flow=main,
#    name="model_inference"
#    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
#    flow_runner=SubprocessFlowRunner(),
#    tags=["cygnss"]
#)

