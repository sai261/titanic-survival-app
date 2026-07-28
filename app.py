import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🚢 Titanic Survival Prediction App")

try:
    with open('logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ 'logistic_model.pkl' not found! Make sure you exported it from your notebook and uploaded it to GitHub.")
    model = None

def user_input_features():
    SEX = st.sidebar.selectbox('Sex', ('1', '0'))
    PC = st.sidebar.selectbox('Pclass', (1, 2, 3))
    Age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=28)
    Fare = st.sidebar.number_input('Fare', min_value=0.0, value=32.0)
    Embarked = st.sidebar.selectbox('Embarked', ('C', 'Q', 'S'))
    Family = st.sidebar.number_input('Siblings/Spouse', min_value=0, value=0)
    Parents = st.sidebar.number_input('Parents', min_value=0, value=0)
    
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    embarked_encoded = embarked_mapping[Embarked]
    
    data = {
        'Sex': int(SEX),  # Convert string '1' or '0' to actual integer 1 or 0
        'Pclass': PC,
        'Age': Age,
        'Fare': Fare,
        'Embarked': embarked_encoded, 
        'SibSp': Family,
        'Parch': Parents 
    }
    features = pd.DataFrame(data, index=[0])
    return features 

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

try:
    TitanicTrain = pd.read_csv("./Titanic_train.csv")
    TitanicTest = pd.read_csv("./Titanic_test.csv")
except FileNotFoundError:
    pass 

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
