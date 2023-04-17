#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import json
import datetime
from datetime import timedelta

import streamlit as st
from pymongo import MongoClient, errors
from PIL import Image
import requests
from io import BytesIO



def user_input_features():
    option = st.sidebar.selectbox(
    'What would you like to see?', ('Results', 'About us'))
    date_ = st.sidebar.date_input("For which date you want to see the results", datetime.date.today() - timedelta(days=12), min_value = datetime.date(2021,1,1), max_value = datetime.date.today() - timedelta(days=12))
    
    
    return date_, option

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    client = MongoClient('mongodb://root:example@mongodb:27017/')
    return client


@st.experimental_memo(ttl=600)
def get_data(date_):
    cygnss = client.cygnss
    from_date = date_
    criteria = {"event_date": {"$eq": from_date}}
    items = cygnss.cygnss_collection.find(criteria)
    items = list(items)  # make hashable for st.experimental_memo
    return items


date_, option = user_input_features()


# Pull data from the collection.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
    # Initializing connection
client = init_connection()

date_ = date_.strftime("%Y-%m-%d")

# drop database if exists, just to not clutter it with multiple values of data
# client.drop_database('cygnss')
items = get_data(date_)

if option == 'About us':
    
    
    st.write("""
        # About US""")

    st.write("The objective of this website is to use a pre-trained CyGNSSnet \
              to predict global ocean wind speed in near time. The results are shown on a web interface, \
              which provides different illustrations of the predicted wind speed and its error compared to ERA5 windspeed data.\
              CyGNSSnet is a neural net developed to predict wind speed from CYGNSS(Cyclone Global Navigation Satellite System) data.\
              The code for CyGNSSnet itself is not public. For more information or if you need to access it contact Caroline Arnold (arnold@dkrz.de)\
              or the Helmholtz AI consultant team for Earth and Environment (consultant-helmholtz.ai@dkrz.de). For more information on CyGNSSnet,\
              see Asgarimehr et al, Remote Sensing of Environment (2022)")

if option == 'Results':


# Display results.
    if len(items) == 0:
        st.write(f" Data does not exist for this date. Choose a different date please!")

    else:
        # Creating UI
        # st.subheader('User Input parameters')

        st.write("""
        # Results """)

        # app heading
        st.write("""
        # Ocean Wind Speed""")
        
        st.write('Date:', date_)

        
        y_bins = ["up to 4m/s", "up to 8m/s", "up to 12m/s",
                "up to 16m/s", "up to 20m/s", "up to 100m/s"]
        for item in items:  # @harsh can this be more than 1 item?
                st.write(f"Total RMSE is: {item['rmse']:.3f} m/s ")
                d = {'Windspeed': y_bins, 'RMSE': item['bin_rmse'], 'Bias': item['bin_bias'],
                'Counts': [int(i) for i in item['bin_counts']]}
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

        for item in items:
                #response = requests.get(item['image_url'])
                # Image.open(BytesIO(response.content))
                scatter = Image.open(item['scatterplot_path'])
                st.markdown(f"## Scatterplot: ERA5 wind speed - model prediction")
                st.image(scatter, caption="Scatterplot")

                histo = Image.open(item['histogram_path'])
                st.markdown(f"## Histogram: ERA5 wind speed and predicted wind speed")
                st.image(histo, caption="Histogram")

                #era_avg = Image.open(item['era_average_path'])
                # st.markdown(f"## ERA 5 Average")
                #st.image(era_avg, caption="ERA5 average")

                #rmse_avg = Image.open(item['rmse_average_path'])
                # st.markdown(f"## RMSE Average")
                #st.image(rmse_avg, caption="RMSE average")

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

