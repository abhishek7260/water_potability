import pickle
import streamlit as st
import numpy as np

# Load the pickled model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        with open('water_quality_model1.pkl', 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully.")  # Debugging statement
        return model
    except FileNotFoundError:
        st.error("Error: Model file not found.")
    except Exception as e:
        st.error("Error loading the model: {}".format(e))

model = load_model()

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
    pH = st.text_input('pH Value')

# Rest of the input fields...

# Default value for quality_prediction
quality_prediction = ''

# Prediction button
if st.button('Predict'):
    try:
        # Validate inputs and convert to float
        features = np.array([float(pH), ...])  # Convert other inputs similarly
        
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
