import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Viz Demo")

#Open the data data file 
with open('RealEstate_Project\df.pkl', 'rb') as file:
    df = pickle.load(file)

#Open the pipeline file 
with open('RealEstate_Project\pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

#Header 
st.header("Enter You Inputs")

### Property type

property_type =  st.selectbox('Property Type', ['flat', 'house'])

#sector
sector = st.selectbox('Sector', sorted( df['sector'].unique().tolist()))

#bedRoom
bedRoom = float(st.selectbox('No of BedRooms', sorted( df['bedRoom'].unique().tolist())))

#bathRoom
bathRoom = float(st.selectbox('No of BathRooms', sorted( df['bathroom'].unique().tolist())))

#balcony
balcony = st.selectbox('No of Balconies', sorted( df['balcony'].unique().tolist()))

#Propery age 
property_age = st.selectbox('Property Age', sorted( df['agePossession'].unique().tolist()))

#Built up area 
built_up_area = float(st.number_input('Built Up Area'))

#Savant Room  
savant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))

#Store Room  
store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

#Furnishing Type
furnishing_type = st.selectbox('Furnishing Type', sorted( df['furnishing_type'].unique().tolist()))

#Luxury Type
luxury_category = st.selectbox('Luxury Type', sorted( df['luxury_category'].unique().tolist()))

#Floor Category
floor_category = st.selectbox('Floor Category', sorted( df['floor_category'].unique().tolist()))


if st.button('predict'):
    # From Dataframe
    data = [[property_type, sector, bedRoom, bathRoom, balcony, property_age, built_up_area, savant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
       'agePossession', 'built_up_area', 'servant room', 'store room',
       'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    # Predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # Display
    st.text("The Price Of The Flat/House Is Between {} cr and {} cr".format(round(low, 2), round(high, 2)))