import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction App")

try:
    with open('logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ 'logistic_model.pkl' not found! Please check your repository.")
    model = None

def user_input_features():
    PC = st.sidebar.selectbox('Pclass', (1, 2, 3), index=2)
    Age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=28)
    Family = st.sidebar.number_input('Siblings/Spouse (SibSp)', min_value=0, value=0)
    Parents = st.sidebar.number_input('Parents (Parch)', min_value=0, value=0)
    Fare = st.sidebar.number_input('Fare', min_value=0.0, value=32.0)

    data = {
        'Pclass': PC,
        'Age': Age,
        'SibSp': Family,
        'Parch': Parents,
        'Fare': Fare
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

if model is not None:
    if st.button("Predict Survival"):
        raw_features = df.values
        
        prediction = model.predict(raw_features)
        probability = model.predict_proba(raw_features)

        st.subheader('Prediction Results')
        survival_prob = probability[0][1]
        st.write(f"Survival Probability: **{survival_prob:.2%}**")

        if prediction[0] == 1:
            st.success("The model predicts this passenger would **Survive**!")
        else:
            st.error("The model predicts this passenger would **Not Survive**.")
