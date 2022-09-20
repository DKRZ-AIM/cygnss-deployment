# cygnss-deployment

Deploy a pre-trained CyGNSSnet to predict global ocean wind speed in near time.

## Getting started

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

