import streamlit as st
import numpy as np
import pickle
from streamlit_option_menu import option_menu

with st.sidebar:
    selected=option_menu('Weather Predicting System  ',['Random Forest Model','Logistic Regression  Model','ANN Model'],default_index=0)
prediction=''
def weather_predictionAnn(input_data):
    import numpy as np

    # Assuming 'input_data' is a list of float values
    float_features = [float(x) for x in input_data]

    features = [np.array(float_features)]

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(features)
    

    # Convert input_data to a numpy array and reshape it to (1, -1)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # Assuming 'loaded_model' is your trained ANN model
    prediction = loaded_model.predict(input_data_as_numpy_array)

    # Get the predicted class index with the highest probability
    print(prediction)
    if prediction.any() == 0:
        return 'Weather is Partly Cloudy'
    elif prediction.any() == 1:
        return 'Weather is Mostly Cloudy'
    else:
        return 'Weather is something other than partly and mostly cloudy'


   
   
def weather_prediction(input_data):
    float_features = [float(x) for x in input_data]

    features = [np.array(float_features)]

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(features)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'Weather is Partly Cloudy'
    if (prediction[0] == 1):
      return 'Weather is Mostly Cloudy'
    else:
      return 'Weather is something other than partly and mostly cloudy'
   
  
if selected=='Random Forest Model':
    st.title('Random Forest Model')
    st.title('Weather Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Precip_Type = st.selectbox("precip Type",('rain','snow'))
    if (Precip_Type=="rain"):
       Precip_Type="0"
    if (Precip_Type=="snow"):
       Precip_Type="1"
    Temperature = st.text_input('Temperature')
    Humidity = st.text_input('Humidity')
    Wind_Speed = st.text_input('Wind Speed')
    Wind_Bearing = st.text_input('Wind Bearing')
    Visibility = st.text_input('Visibility')
    Pressure = st.text_input('Pressure')
    month = st.text_input('month')
    hour = st.text_input('hour')
    
    
    # code for Prediction
    prediction = ''
    
    # creating a button for Prediction
    
    if st.button('Weather Predicton Result'):
        loaded_model = pickle.load(open("C:/Users/HP/Downloads/rf_model1.sav", 'rb'))
        
        prediction = weather_prediction([Precip_Type, Temperature, Humidity, Wind_Speed, Wind_Bearing, Visibility, Pressure, month,hour])
        
        
    st.success(prediction)
    
  
    
    
    
    
if selected=='Logistic Regression  Model':
    st.title('Logistic Regression  Model')
    st.title('Weather Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Precip_Type = st.selectbox("precip Type",('rain','snow'))
    if (Precip_Type=="rain"):
       Precip_Type="0"
    if (Precip_Type=="snow"):
       Precip_Type="1"
    Temperature = st.text_input('Temperature')
    Humidity = st.text_input('Humidity')
    Wind_Speed = st.text_input('Wind Speed')
    Wind_Bearing = st.text_input('Wind Bearing')
    Visibility = st.text_input('Visibility')
    Pressure = st.text_input('Pressure')
    month = st.text_input('month')
    hour = st.text_input('hour')
    
    
    # code for Prediction
    prediction = ''
    
    # creating a button for Prediction
    
    if st.button('Weather Predicton Result'):
        loaded_model = pickle.load(open("C:/Users/HP/Downloads/lg_model1.sav", 'rb'))
        
        prediction = weather_prediction([Precip_Type, Temperature, Humidity, Wind_Speed, Wind_Bearing, Visibility, Pressure, month,hour])
        
        
    st.success(prediction)
    
    
    

if selected=='ANN Model':
    st.title('ANN Model')
    st.title('Weather Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Precip_Type = st.selectbox("precip Type",('rain','snow'))
    if (Precip_Type=="rain"):
       Precip_Type="0"
    if (Precip_Type=="snow"):
       Precip_Type="1"
    Temperature = st.text_input('Temperature')
    Humidity = st.text_input('Humidity')
    Wind_Speed = st.text_input('Wind Speed')
    Wind_Bearing = st.text_input('Wind Bearing')
    Visibility = st.text_input('Visibility')
    Pressure = st.text_input('Pressure')
    month = st.text_input('month')
    hour = st.text_input('hour')
    
    
    # code for Prediction
    prediction = ''
    
    # creating a button for Prediction
    
    if st.button('Weather Predicton Result'):
        loaded_model = pickle.load(open("C:/Users/HP/Downloads/ann_model3.sav", 'rb'))
        
        prediction = weather_predictionAnn([Precip_Type, Temperature, Humidity, Wind_Speed, Wind_Bearing, Visibility, Pressure, month,hour])
        
        
    st.success(prediction)
  


# creating a function for Prediction

    
  
