'''


from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
'''
import streamlit as st; from PIL import Image
import sklearn
from sklearn.ensemble         import RandomForestRegressor
import joblib
import pandas as pd
import numpy as np

# Load models
#RF = pickle.load(open('LR.pkl','rb'))
RF = joblib.load('RF.joblib')
#RF = pickle.load(open('RF_new.pkl','rb'))
DTR = joblib.load('DTR.joblib')
#DTR = pickle.load(open('DTR.pkl','rb'))
SVR = joblib.load('SVR.joblib')
#SVR = pickle.load(open('SVR.pkl','rb'))
GBR = joblib.load('GBR.joblib')
#GBR = pickle.load(open('GBR.pkl','rb'))
KNN = joblib.load('KNN.joblib')
#KNN = pickle.load(open('KNN.pkl','rb'))
ETR = joblib.load('ETR.joblib')
#ETR = pickle.load(open('ETR.pkl','rb'))
BG = joblib.load('BG.joblib')
#BG = pickle.load(open('BG.pkl','rb'))
ADA = joblib.load('ADA.joblib')
#ADA = pickle.load(open('ADA.pkl','rb'))
XGB = joblib.load('XGB.joblib')
#XGB = pickle.load(open('XGB.pkl','rb'))
MLP = joblib.load('MLP.joblib')
#MLP = pickle.load(open('MLP.pkl','rb'))
LR = joblib.load('LR.joblib')
#LR = pickle.load(open('LR.pkl','rb'))


st.write('Geopolymer Concrete Compressive Strength predictor:')

# Dropdown menu for model selection
selected_model = st.selectbox(
    'Select ML model to predict:',
    ('RF_model', 'DTR_model', 'SVR_model', 'GBR_model', 'KNN_model', 'ETR_model', 'BG_model', 'ADA_model', 'XGB_model', 'MLP_model', 'LR_model')
)

# Input fields
Fine_aggregates_content = st.number_input('Fine aggregate content in kg/m3')
Coarse_aggregates_content = st.number_input('Coarse aggregate content in kg/m3')
Molarity = st.number_input('Molarity of NaoH solution')
Sodium_hydroxide_content = st.number_input('Sodium hydroxide solution content in kg/m3')
Sodium_silicate_content = st.number_input('Sodium silicate solution content in kg/m3')
Fly_ash_content = st.number_input('Fly Ash content in kg/m3')
Water_content = st.number_input('Water content in kg/m3')
Superplasticizer_content = st.number_input('Superplasticizer content in kg/m3')
Temperature = st.number_input('Curing temperature in degree centigrates')
Curing_age = st.number_input('Curing age in days')

        
input1 = [Fine_aggregates_content, Coarse_aggregates_content, Molarity, Sodium_hydroxide_content, Sodium_silicate_content, Fly_ash_content, Water_content, Superplasticizer_content, Temperature, Curing_age]
input1 = np.array(input1).reshape(1, -1)

if st.button('Predict Compressive Strength'):    
  if selected_model == 'RF_model':
    CS = RF.predict(input1)
    st.write("Predicted compressive strength is:" + str(CS) + " MPa")
  elif selected_model == 'DTR_model':
    CS = DTR.predict(input1)
    st.write("Predicted compressive strength is:" + str(CS) + " MPa")
  elif selected_model == 'SVR_model':
    CS = SVR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'GBR_model':
    CS = GBR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'KNN_model':
    CS = KNN.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'ETR_model':
    CS = ETR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'BG_model':
    CS = BG.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'ADA_model':
    CS = ADA.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'XGB_model':
    CS = XGB.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'MLP_model':
    CS = MLP.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  elif selected_model == 'LR_model':
    CS = LR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
  else:
    st.write("Error in inputs: Compressive strength could not be predicted.")

st.write("Note: The predicted values are based on machine learning models developed by the author. "
         "The research is in initial stages and the values are just for giving a rough idea of the properties of Fly ash based geopolymer concrete. "
         "The results shall not be considered as final and experimental assessment of properties shall be done in practice.")

image1 = Image.open('developed_comb.png')
st.image(image1)
