import pickle
import streamlit as st
import numpy as np

# Load the pickled model
@st.cache_resource
def load_model():
    try:
        with open('water_quality_model1.pkl', 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully.")  # Debugging statement
        return model
    except FileNotFoundError:
        st.error("Error: Model file not found.")
   

model = load_model()

# Define a function to make predictions
def predict_water_quality(features):
    try:
        prediction = model.predict(features.reshape(1, -1))
        return prediction[0]
    except Exception as e:
        st.error("Error predicting water quality: {}".format(e))

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

# Default value for quality_prediction
quality_prediction = ''

# Prediction button
if st.button('Predict'):
    try:
        # Validate inputs and convert to float
        features = np.array([float(pH), float(Hardness), float(Solids), float(Chloramines),
                             float(Sulfate), float(Conductivity), float(Organic_carbon),
                             float(Trihalomethanes), float(Turbidity)])

        # Make prediction
        prediction = predict_water_quality(features)

        # Display prediction result
        if prediction == 1:
            quality_prediction = 'The water is safe to drink.'
        else:
            quality_prediction = 'The water is not safe to drink.'
    except ValueError:
        st.error("Error: Please enter valid numeric values.")

# Display the prediction result
st.success(quality_prediction)
