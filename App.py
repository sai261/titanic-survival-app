import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🚢 Titanic Survival Prediction App")

try:
    with open('logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ 'logistic_model.pkl' not found! Please check your file path on GitHub.")
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
        'Pclass': PC,
        'Sex': int(SEX), 
        'Age': Age,
        'SibSp': Family,
        'Parch': Parents,
        'Fare': Fare,
        'Embarked': embarked_encoded
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
    if hasattr(model, 'feature_names_in_'):
        df = df[model.feature_names_in_]

    if st.button("Predict Survival"):
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1] # Get survival probability
        
        st.subheader('Prediction Results')
        st.write(f"Survival Probability: **{probability:.2%}**")
        
        if prediction[0] == 1:
            st.success("🎉 Passenger is predicted to Survive!")
        else:
            st.error("💀 Passenger is predicted Not to Survive.")

  
