# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\active
# pip install streamlit pandas numpy seaborn  matplotlib scikit-learn
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title("Insurance Premium Price Prediction App")

# try loading model safely
try:
    model = pickle.load(open('model_gb.pkl', 'rb'))
except:
    st.error("Model file not found! Please check model_gb.pkl")
    st.stop()

scaler = MinMaxScaler()

# inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
gender = st.selectbox('Gender', ('male', 'female'))
bmi = st.number_input('BMI', min_value=10.0, max_value=100.0, value=30.0)
smoker = st.selectbox('Smoker', ('yes', 'no'))
children = st.number_input('Number of Children', min_value=0, max_value=10, value=2)
region = st.selectbox('Region', ('southwest', 'southeast', 'northwest', 'northeast'))

# encoding
Smoker = 1 if smoker == 'yes' else 0
sex_female = 1 if gender == 'female' else 0
sex_male = 1 if gender == 'male' else 0

region_dict = {'southwest': 0, 'southeast': 3, 'northwest': 1, 'northeast': 2}
Region = region_dict[region]

# dataframe
input_features = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'Smoker': [Smoker],
    'sex_female': [sex_female],
    'sex_male': [sex_male],
    'Region': [Region]
})

# scaling
input_features[['age', 'bmi']] = scaler.fit_transform(input_features[['age', 'bmi']])

# prediction
if st.button('Predict'):
    try:
        prediction = model.predict(input_features)
        output = round(np.exp(prediction[0]), 2)
        st.success(f"Price Prediction: ${output}")
    except Exception as e:
        st.error(f"Error: {e}")