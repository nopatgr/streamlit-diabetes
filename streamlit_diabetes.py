import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
diabetes = pd.read_csv('diabetes.csv')

# Define features and target variable
X = diabetes.iloc[:, :-1].values
y = diabetes.iloc[:, -1].values

# Fit linear regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Define function to make predictions
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    prediction = regressor.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    return prediction

# Create Streamlit app
st.title('Prediksi Diabetes Menggunakan Regresi Linear')

# Define input fields
pregnancies = st.number_input('Jumlah Kehamilan')
glucose = st.number_input('Glukosa')
blood_pressure = st.number_input('Tekanan Darah')
skin_thickness = st.number_input('Ketebalan Kulit')
insulin = st.number_input('Insulin')
bmi = st.number_input('Indeks Massa Tubuh')
diabetes_pedigree_function = st.number_input('Fungsi Pedigree Diabetes')
age = st.number_input('Umur')

# Make prediction
if st.button('Prediksi Diabetes'):
    result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    st.success('Hasil Prediksi: {}'.format(result[0]))

    