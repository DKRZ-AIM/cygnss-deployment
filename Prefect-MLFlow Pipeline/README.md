# Prefect-MLFlow Pipeline

Pipeline => Download the data, Train a model from scratch, evaluate it and log the model and scores to MLFlow

Run this pipeline periodically and only if the model performance degrades then train the model else only evaluate the model and log the  performance
Deploy your best model to MLFlow and use it as an API

TODO: handle failures

## Serving Model

```
mlflow models serve --model-uri runs:/bf1847fc9369486da1cb6191059d4c5c/model

general:

mlflow models serve --model-uri runs:/{run_id}/model
```

- Model Inference: Make sure 5000 port is open. Then, open up a new terminal and activate a conda env which you were using before and run following command:

```
curl -d '{"data":[[0.178,0.0,4.05,0.0,0.51,6.416,84.1,2.6463,5.0,296.0,16.6]]}' -H 'Content-Type: application/json'  localhost:5000/invocations
```


## Running the UI part:

Open a terminal with activate conda env

1. pip install streamlit

2. RUN  jupyter-notebook, it will create mlflow folder and log the model and params

3. To run MLFLOW Server:,  use following command, this is to change the state of the model from stage to production:

	mlflow server -p 5002 --backend-store-uri sqlite:///mlruns.db  --default-artifact-root s3://my-mlflow-bucket/

4. Then we can Serve the above registered model by running following: ( you can either put this in a bash file and run bash serve_ml_model.sh), or:

	 export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

	 mlflow models serve -m "models:/ElasticnetWineModel/Production"

  This will run the model as an api, on localhost:5000

5. Run the UI -> Where updated model is getting used:
	streamlit run dashboard.py 

