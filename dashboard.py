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
        date_ = st.sidebar.date_input("For which date you want to see the results", datetime.date.today() - datetime.timedelta(days=10), min_value = datetime.date(2021,1,1), max_value = datetime.date.today() - datetime.timedelta(days=10))
        st.write('', option)
        return date_, option

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
date_, option = user_input_features()
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
if option == 'RMSE':
    y_bins = ["up to 4m/s", "up to 8m/s", "up to 12m/s", "up to 16m/s", "up to 20m/s", "up to 100m/s"]
    for item in items: # @harsh can this be more than 1 item?
        st.write(f"Total RMSE is: {item['rmse']:.3f} m/s ")
        d = {'Windspeed': y_bins, 'RMSE': item['bin_rmse'], 'Bias': item['bin_bias'],\
                'Counts': item['bin_counts'], 'Time': item['time']}
        df = pd.DataFrame(data=d)
        # hide first column (index) of the table
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(data=df)

if option == 'Visualization':
    for item in items:
    #response = requests.get(item['image_url'])
        scatter = Image.open(item['scatterplot_path'])#Image.open(BytesIO(response.content))
        st.markdown(f"## Scatterplot: ERA5 wind speed - model prediction")
        st.image(scatter, caption="Scatterplot")

        histo = Image.open(item['histogram_path'])
        st.markdown(f"## Histogram: ERA5 wind speed and predicted wind speed")
        st.image(histo, caption="Histogram")

        era_avg = Image.open(item['era_average_path'])
        st.markdown(f"## ERA 5 Average")
        st.image(era_avg, caption="ERA5 average")

        rmse_avg = Image.open(item['rmse_average_path'])
        st.markdown(f"## RMSE Average")
        st.image(rmse_avg, caption="RMSE average")

        today_longavg = Image.open(item['today_longrunavg_path'])
        st.markdown(f"## RMSE - Today and Longrun Average")
        st.image(today_longavg, caption="RMSE - Today and Longrun Average")

        today_long_bias = Image.open(item['today_long_bias_path'])
        st.markdown(f"## BIAS - Today and Longrun Average")
        st.image(today_long_bias, caption="Bias - Today and Longrun Average")

        sample_counts = Image.open(item['sample_counts_path'])
        st.markdown(f"## Sample Counts")
        st.image(sample_counts, caption="Sample Counts")

        rmse_bins_era = Image.open(item['rmse_bins_era_path'])
        st.markdown(f"## RMSE for different Windspeed Bins")
        st.image(rmse_bins_era, caption="RMSE for different Windspeed Bins")

        bias_bins_era = Image.open(item['bias_bins_era_path'])
        st.markdown(f"## Bias for different Windspeed Bins")
        st.image(bias_bins_era, caption="Bias for different Windspeed Bins")


