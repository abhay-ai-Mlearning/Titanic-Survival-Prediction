import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Logistic Regression_titanic.pkl")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex_text = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_text == "Male" else 0

age = st.number_input("Age", min_value=0, max_value=100, value=25)

sibsp = st.number_input("Siblings/Spouse", min_value=0, max_value=10, value=0)

parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)

fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)

embarked = st.text_input("Embarked")

alone_text = st.selectbox("Alone", ["False", "True"])
alone = 1 if alone_text == "True" else 0

if st.button("Predict"):

    data = pd.DataFrame([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked,
        alone
    ]], columns=[
        'pclass',
        'sex',
        'age',
        'sibsp',
        'parch',
        'fare',
        'embarked',
        'alone'
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")

