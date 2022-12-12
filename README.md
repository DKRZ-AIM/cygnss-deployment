# Web Interface for Wind Speed Prediction

### About

The objective of this repository is to deploy a pre-trained *CyGNSSnet* to predict global ocean wind speed in near time. The results are shown on a web interface, which provides different illustrations of the predicted wind speed and its error compared to [ERA5 windspeed](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) data. 

*CyGNSSnet* is a neural net developed to predict wind speed from [CYGNSS](https://podaac.jpl.nasa.gov/dataset/CYGNSS_L2_V3.0)(**Cy**clone **G**lobal **N**avigation **S**atellite **S**ystem) data. The code for *CyGNSSnet* itself is not public. For more information or if you need to access it contact Caroline Arnold (arnold@dkrz.de) or the Helmholtz AI consultant team for Earth and Environment (consultant-helmholtz.ai@dkrz.de).   
### Workflow

![Workflow](/Workflow.png)



### Quick start

To start the deployment run ```sh set_up_infrastructure.sh```.

This clones the git repository and starts the deployment using docker-compose.
Make sure you have docker and docker-compose installed. 

If you have already the cloned the git repository move to the directory ```docker_cygnss_deployment``` and run 

```
docker-compose up
``` 

To stop the container, run following command:
```
docker-compose -f ./docker-compose.yml down --remove-orphans
```

Note: In order to run it you need access to the external submodule containing the CyGNSSnet. 

The deployment is scheduled using prefect. It is executed every day and downloads the CyGNSS data for the current date minus 10 days. Then the predictions are calculated, stored in a mongodb database and displayed on a streamlit dashboard.

To access the streamlit dashboard: http://localhost:8501

To access the mongodb database: http://localhost:8081

To access the prefect ui: http://localhost:5000


### Repository Structure

```
API.py: download CyGNSS data
Preprocessing.py: download ERA5 data and preprocess data
dashboard.py: streamlit dashboard
plots.py: helper functions to create the plots for the streamlit dashboard
prefect-deploy.py: Deployment scheduled for every day
externals/: folder with CyGNSSnet code
notebooks/: folder with some notebooks that were created during the development
docker_cygnss_deployment/: folder with docker files to start deployment
```
    

