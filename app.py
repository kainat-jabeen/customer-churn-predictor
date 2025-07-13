import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open("log_model.pkl", "rb"))

le_region = LabelEncoder().fit(['East', 'West', 'North', 'South'])
le_payment = LabelEncoder().fit(['Credit Card', 'PayPal', 'Apple Pay'])

st.title("🛍️ FreshDirect Customer Churn Predictor")

recency = st.slider("Recency (days)", 1, 365)
frequency = st.slider("Order Frequency", 1, 50)
avg_order_value = st.slider("Avg Order Value ($)", 20.0, 200.0)
complaints = st.slider("Complaints", 0, 5)
delivery_issues = st.slider("Delivery Issues", 0, 3)
is_subscribed = st.selectbox("Email Subscribed?", [0, 1])
region = st.selectbox("Region", ['East', 'West', 'North', 'South'])
payment_method = st.selectbox("Payment Method", ['Credit Card', 'PayPal', 'Apple Pay'])

input_df = pd.DataFrame({
    'recency': [recency],
    'frequency': [frequency],
    'avg_order_value': [avg_order_value],
    'complaints': [complaints],
    'delivery_issues': [delivery_issues],
    'is_subscribed': [is_subscribed],
    'region': le_region.transform([region]),
    'payment_method': le_payment.transform([payment_method])
})

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"Likely to churn (probability: {prob:.2f})")
    else:
        st.success(f"✅ Not likely to churn (probability: {prob:.2f})")