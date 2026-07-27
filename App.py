import streamlit as st
import pandas as pd
import pickle

st.title("🚢 Titanic Survival Prediction App")

# Load the pickle model safely
try:
    with open('logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ 'logistic_model.pkl' not found! Make sure it is uploaded to GitHub.")
    model = None

def user_input_features():
    # Only collect inputs for features your model actually trained on
    PC = st.sidebar.selectbox('Pclass (Ticket Class)', (1, 2, 3))
    Age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=28)
    Family = st.sidebar.number_input('Siblings/Spouse', min_value=0, value=0)
    Parents = st.sidebar.number_input('Parents', min_value=0, value=0)
    Fare = st.sidebar.number_input('Fare ($)', min_value=0.0, value=32.0)

    # EXACT 5 features in the EXACT order shown inside your .pkl file
    data = {
        'Pclass': PC,
        'Age': Age,
        'SibSp': Family,
        'Parch': Parents,
        'Fare': Fare
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

if model is not None:
    if st.button("Predict Survival"):
        try:
            # Generate the prediction
            prediction = model.predict(df)
            probability = model.predict_proba(df)[0][1]
            
            st.subheader('Prediction Results')
            st.write(f"Survival Probability: **{probability:.2%}**")
            
            if prediction[0] == 1:
                st.success("🎉 Passenger is predicted to Survive!")
            else:
                st.error("💀 Passenger is predicted Not to Survive.")
        except Exception as e:
            st.error(f"Prediction failed due to structure mismatch: {e}")
