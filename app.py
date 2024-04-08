import pickle
import streamlit as st
import numpy as np
try:
    # Load the pickled model
    with open('water_quality_model1.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: Model file not found.")
except Exception as e:
    print("Error loading the model:", e)
    # Handle other exceptions as needed


# Define a function to make predictions
def predict_water_quality(features):
    prediction = model.predict(features.reshape(1, -1))
    return prediction[0]

# Set up the Streamlit app
st.title('Water Potability Prediction using ML')

# Define columns for input fields
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

with col1:
    pH = st.text_input('pH Value')

with col2:
    Hardness = st.text_input('Hardness Value')

with col3:
    Solids = st.text_input('Solids Value')

with col4:
    Chloramines = st.text_input('Chloramines Value')

with col5:
    Sulfate = st.text_input('Sulfate Value')

with col6:
    Conductivity = st.text_input('Conductivity Value')

with col7:
    Organic_carbon = st.text_input('Organic_carbon Value')

with col8:
    Trihalomethanes = st.text_input('Trihalomethanes Value')

with col9:
    Turbidity = st.text_input('Turbidity Value')

# Default value for price_prediction
quality_prediction = ''

# Prediction button
if st.button('Predict'):
    # Create a feature vector from user input
    features = np.array([pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity], dtype=float)
    
    # Make prediction
    prediction = predict_water_quality(features)
    
    # Display prediction result
    if prediction == 1:
        quality_prediction = 'The water is safe to drink.'
    else:
        quality_prediction = 'The water is not safe to drink.'

# Display the prediction result
st.success(quality_prediction)
