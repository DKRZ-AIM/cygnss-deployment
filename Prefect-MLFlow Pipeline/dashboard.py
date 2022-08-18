#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import json
#app heading
st.write("""
# Wine Quality Prediction App
This app predicts the ***Wine Quality*** type!
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        residual_suagar = st.sidebar.slider('residual_suagar', 0.01,0.6 , 0.08)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
        free_sulfur_dioxide=st.sidebar.slider('free sulfur dioxide', 6.0,289.0 , 46.0)
        density=st.sidebar.slider('density', 8.4,14.9, 10.4)
        pH=st.sidebar.slider('pH', 8.4,14.9, 10.4)
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
                'residual_suagar': residual_suagar,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'free_sulfur_dioxide':free_sulfur_dioxide,
              'density':density,
              'pH': pH,    
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features



def get_model_predictions(data):
    """
    Get model predictions  
    ENDPOINT = Calls an endpoint to get the predictions
    REGISTRY = Loads model from registry and predicts
    MOCKED = Randomly generated prediction
    """
    option="ENDPOINT"

    if option == "ENDPOINT":
        # Currently not supported for multi-input models
        DEPLOYED_ENDPOINT = "http://127.0.0.1:5000/invocations"
        headers = {"Content-Type":"application/json"}

        to_send_data_json = json.dumps({"data":data})                
        prediction = requests.post(url=DEPLOYED_ENDPOINT, 
                                   data = to_send_data_json, headers=headers)
        
        prediction = prediction.content.decode("utf-8")
        print(prediction)
    
    if option == "REGISTRY":
        # Currently not supported for multi-input models
        model_name = "ElasticnetWineModel"
        model_version = "25"
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        
        prediction = model.predict(data)
        
    return prediction

st.subheader('User Input parameters')
df = user_input_features()
st.write(df)
#reading csv file
data=pd.read_csv("winequality-red.csv")

#random forest model
st.subheader('Wine quality labels and their corresponding index number')
st.write(pd.DataFrame({
   'wine quality': [3, 4, 5, 6, 7, 8 ]}))


prediction = get_model_predictions(df.values.tolist())

st.subheader('Prediction')
st.write(prediction)

