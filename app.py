import pickle
import streamlit as st
import numpy as np

# Load the pickled model
model = None
try:
    with open('water_quality_model1.sav', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file not found.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

if model is not None:
    # Define a function to make predictions
    def predict_water_quality(features):
        prediction = model.predict(features.reshape(1, -1))
        return prediction[0]

    # Set up the Streamlit app
    st.title('Water Potability Prediction using ML')

    # Define columns for input fields
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    with col1:
        pH = st.number_input('pH Value', min_value=0.0, max_value=14.0, step=0.1)

    with col2:
        Hardness = st.number_input('Hardness Value', min_value=0.0)

    with col3:
        Solids = st.number_input('Solids Value', min_value=0.0)

    with col4:
        Chloramines = st.number_input('Chloramines Value', min_value=0.0)

    with col5:
        Sulfate = st.number_input('Sulfate Value', min_value=0.0)

    with col6:
        Conductivity = st.number_input('Conductivity Value', min_value=0.0)

    with col7:
        Organic_carbon = st.number_input('Organic_carbon Value', min_value=0.0)

    with col8:
        Trihalomethanes = st.number_input('Trihalomethanes Value', min_value=0.0)

    with col9:
        Turbidity = st.number_input('Turbidity Value', min_value=0.0)

    # Default value for quality_prediction
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
else:
    st.error("Error: Model not loaded. Please check the model file.")
