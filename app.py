import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# load the scalers and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", 18, 95)
tenure = st.slider("Tenure", 0, 10)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
number_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


# concat
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# When the user clicks Predict, scale and predict
if st.button("Predict Churn"):
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    # Make prediction
    prediction = model.predict(input_data_scaled)
    churn_probability = float(prediction[0][0])
    if churn_probability > 0.5:
        st.write("The customer is likely to churn (probability={:.2f})".format(churn_probability))
    else:
        st.write("The customer is not likely to churn (probability={:.2f})".format(churn_probability))
