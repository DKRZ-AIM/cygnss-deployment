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


# Initialize connection.
# Uses st.experimental_singleton to only run once.

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

@st.experimental_singleton
def init_connection(): 
        client = MongoClient(host='localhost:27017' , serverselectiontimeoutms=3000)
        return client



def write_data():
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
client.drop_database('cygnss')

# write data
write_data()

# Fetching data
items = get_data(date_)


st.subheader('Results')
# Print results.
for item in items:
        st.write(f"RMSE is: {item['rmse']} ")
        response = requests.get(item['image_url'])
        image = Image.open(BytesIO(response.content))

        
        #displaying the image on streamlit app
        st.image(image, caption='Visualization')
