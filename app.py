import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

preg = PolynomialFeatures(degree=4)

model = pickle.load(open('Salary_model.sav', 'rb'))
df = pd.read_csv('Salary.csv')
st.set_page_config(page_title='Salary Prediction Model', page_icon="./favicon.ico")
st.title('Salary Prediction Model')
st.subheader('Experience vs Salary')
st.write('This model uses polynomial regression to predict salaries according to years of experience.')
st.image('./img.jpg')
st.header('Years of Experience')
exp=st.number_input('', 1, 15)
exp_pred = preg.fit_transform([[exp]])
sal = model.predict(exp_pred)
st.header('Predicted Salary:')
st.subheader("Rs: "+str(np.round(sal[0], 2)))


