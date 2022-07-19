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

