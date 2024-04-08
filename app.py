import streamlit as st
import pickle
import numpy as np

# Load the pickled model
def load_model():
    try:
        with open('water_quality_model1.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: Model file not found.")
    except Exception as e:
        st.error("Error loading the model: {}".format(e))

# Load the model
model = load_model()

# Display success message
if model:
    st.success("Model loaded successfully.")

# Define a function to make predictions
def predict_water_quality(features):
    try:
        if model is not None:
            prediction = model.predict(features.reshape(1, -1))
            return prediction[0]
        else:
            st.error("Model is not loaded.")
    except Exception as e:
        st.error("Error predicting water quality: {}".format(e))

# Set up the Streamlit app
st.title('Water Potability Prediction using ML')

# Define columns for input fields
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

with col1:
    pH = st.number_input('pH Value')

with col2:
    Hardness = st.number_input('Hardness Value')

with col3:
    Solids = st.number_input('Solids Value')

with col4:
    Chloramines = st.number_input('Chloramines Value')

with col5:
    Sulfate = st.number_input('Sulfate Value')

with col6:
    Conductivity = st.number_input('Conductivity Value')

with col7:
    Organic_carbon = st.number_input('Organic_carbon Value')

with col8:
    Trihalomethanes = st.number_input('Trihalomethanes Value')

with col9:
    Turbidity = st.number_input('Turbidity Value')

# Default value for quality_prediction
quality_prediction = ''

# Prediction button
if st.button('Predict'):
    try:
        # Validate inputs
        if not (0 <= pH <= 14):
            st.error("Please enter a valid pH value (between 0 and 14).")
        else:
            # Convert inputs to float
            features = np.array([
                float(pH), float(Hardness), float(Solids), float(Chloramines),
                float(Sulfate), float(Conductivity), float(Organic_carbon),
                float(Trihalomethanes), float(Turbidity)
            ])

            # Make prediction
            prediction = predict_water_quality(features)

            # Display prediction result
            if isinstance(prediction, (int, float)):
                if prediction == 1:
                    quality_prediction = 'The water is safe to drink.'
                else:
                    quality_prediction = 'The water is not safe to drink.'
            else:
                quality_prediction = f"Predicted Water Quality Index: {prediction:.2f}"

    except ValueError:
        st.error("Error: Please enter valid numeric values.")

# Display the prediction result
st.success(quality_prediction)
