import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and label encoder
model = joblib.load('best_model.pkl')
encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Titanic Survival Predictor")
st.title(" Titanic Survival Prediction App")

st.markdown("Enter the passenger details below to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, step=1)
fare = st.number_input("Passenger Fare", min_value=0.0, step=0.1)

# Encode 'sex'
sex_encoded = encoder.transform([sex])[0]

# Predict button
if st.button("Predict Survival"):
    user_data = np.array([[pclass, age, sex_encoded, sibsp, parch, fare]])
    prediction = model.predict(user_data)[0]
    
    if prediction == 1:
        st.success(" This passenger **would have survived**!")
    else:
        st.error(" Unfortunately, this passenger **would not have survived**.")
