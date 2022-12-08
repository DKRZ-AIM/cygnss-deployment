# Web Interface for Wind Speed Prediction

## About

The objective of this repository is to deploy a pre-trained *CyGNSSnet* to predict global ocean wind speed in near time. The results are shoen on a web interface, which provides different illustrations of the predicted wind speed and its error compared to [ERA5 windspeed](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) data. 

*CyGNSSnet* is a neural net developed to predict wind speed from [CYGNSS](https://podaac.jpl.nasa.gov/dataset/CYGNSS_L2_V3.0)(**Cy**clone **G**lobal **N**avigation **S**atellite **S**ystem) data. The code for *CyGNSSnet* itself is not public. For more information or if you need to access it contact Caroline Arnold (arnold@dkrz.de) or the Helmholtz AI consultant team for Earth and Environment (consultant-helmholtz.ai@dkrz.de).   

## Getting started

In order to start the deployment ...

#### Working: This code will run prefect tasks. And, resulting data will be saved in database. Then, from updated database, data will be shown in streamlit dashboard to the user.

# Installation
- Folder structure: All folders in same directory
    ```
    2020-03-gfz-remote-sensing/  (Cygnss code)
    2022-cygnss-deployment/   (data)
    cygnss-deployment/    (current git repo)
    ```
- Make sure you have docker and docker-compose installed 
- Go to cygnss-deployment folder/docker_cygnss_deployment
- Run the following command and it will install all the dependency and will fire all the necessary servers (MLFlow, Mongo-DB and Streamlit): 
    ```
    docker-compose -f ./docker-compose.yml up 
    ```
- Access For the servers on localhost
    ```
      streamlit-dashboard: http://localhost:8501/
      mongo-express-Dashboard: http://localhost:8081/    
    ```
    
- To stop the container, run following command:
    ```
    docker-compose -f ./docker-compose.yml down --remove-orphans
    ```

