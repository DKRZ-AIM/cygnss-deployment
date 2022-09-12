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


def calculate_rmse():
    
    filename = "best_predictions.h5"

    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        
        # Use True lable to calculate RMSE
        return 0.7

@task
def is_model_degraded_condition():
    rmse = calculate_rmse()
    if rmse > 0.4:
        return True
    else:
        return False  

    

    
@task
def write_data(client):
        data_1 = {
        "rmse": 3.1, 
        "event_date":  datetime.datetime(2022, 8, 10),
        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
        }

        data_2 = {
        "rmse": 2.1,         
        "event_date":  datetime.datetime(2022, 8, 9),
        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
        }

        data_3 = {
        "rmse": 3.2,         
        "event_date":  datetime.datetime(2022, 8, 8),
        "image_url": "https://www.dkrz.de/en/about-en/aufgaben/dkrz-and-climate-research/@@images/image/large"
        }


        cygnss_collection = client["cygnss"].cygnss_collection

        cygnss_collection = cygnss_collection.insert_many([data_1, data_2, data_3])

        print(f"Multiple tutorials: {cygnss_collection.inserted_ids}")

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

    # TODO
    # download_data()
    
    # TODO
    # pre_process()

    model_path = '/work/ka1176/shared_data/2022-cygnss-deployment/'\
            'cygnss_trained_model/ygambdos_yykDM/checkpoint'
    model = 'cygnssnet-epoch=0.ckpt'
    data_path = '../../shared_data/2022-cygnss-deployment/small_data/' #'../data'
    h5_file = h5py.File(os.path.join(data_path, 'test_data.h5'), 'r', rdcc_nbytes=0)

    mlflow.set_tracking_uri("sqlite:///mlruns.db") # future todo: change this to other db
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
    
    # save results in db    
    client = mongo_db_connection()
    drop_database(client)
    write_data(client)
    get_data(client)


main()

# DeploymentSpec(
#    flow=main,
#    name="model_inference"
#    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
#    flow_runner=SubprocessFlowRunner(),
#    tags=["cygnss"]
#)

