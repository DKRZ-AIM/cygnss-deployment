#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import json
import datetime
import streamlit as st
from pymongo import MongoClient, errors
from PIL import Image 
import requests
from io import BytesIO



#app heading
st.write("""
# Wind Speed""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')

def user_input_features():
        option = st.sidebar.selectbox('What would you like to see?', ('RMSE', 'Visualization'))
        date_ = st.sidebar.date_input("For which date you want to see the results", datetime.date.today() - datetime.timedelta(days=5), min_value = datetime.date(2021,1,1), max_value = datetime.date.today() - datetime.timedelta(days=5))
        st.write('', option)
        return date_

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():         
        client = MongoClient('mongodb://root:example@mongodb:27017/')
        return client

# Pull data from the collection.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def get_data(date_):        
        cygnss = client.cygnss        
        from_date = datetime.datetime.combine(date_, datetime.time())
        criteria = {"event_date": {"$eq": from_date}}
        items = cygnss.cygnss_collection.find(criteria)        
        items = list(items)  # make hashable for st.experimental_memo
        return items


# Creating UI
st.subheader('User Input parameters')
date_ = user_input_features()
st.write('Date:', date_)

# Initializing connection
client = init_connection()


# drop database if exists, just to not clutter it with multiple values of data
# client.drop_database('cygnss')

# Fetching data
items = get_data(date_)

st.subheader('Results')

# Display results.
if len(items) == 0:
        st.write(f" Data does not exist for this date. Choose a different date please!")
for item in items:
        st.write(f"RMSE is: {item['rmse']} ")
        response = requests.get(item['image_url'])
        image = Image.open(BytesIO(response.content))

        
        #displaying the image on streamlit app
        st.image(image, caption='Visualization')
