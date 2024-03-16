import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn import metrics
import math

from sklearn.metrics import r2_score
from sklearn.svm import SVR
from tensorflow.python import keras
from keras.layers import Dense
from keras.models import Sequential
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from tensorflow.python.distribute import input_lib

st.set_page_config(page_title="sup" , page_icon=":bar_chart:" ,layout="wide")
st.write("""
# Predict The Price of Your Dream House
""")
st.write('___')
df= pd.read_csv("HousePrice.csv")
st.sidebar.header("Choose your filter: ")


df=df.dropna().reset_index(drop=True)
df=df[['Area','Room','Parking','Warehouse','Elevator','Address','PriceMC']]

percentile25 = df['PriceMC'].quantile(0.25)
percentile75 = df['PriceMC'].quantile(0.75)
area25=df['Area'].quantile(0.25)
area75=df['Area'].quantile(0.75)
iqr=percentile75 -percentile25
iqrArea=area75 -area25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
upper_limit_area = area75 + 1.5 * iqrArea
lower_limit_area = area25 - 1.5 * iqrArea
# df[df['PriceMC'] > upper_limit]
# df[df['PriceMC'] < lower_limit]

new_df =df[df['PriceMC'] < upper_limit]
# new_df =df[df['Area'] < upper_limit_area]

new_df=new_df[new_df['PriceMC'] > lower_limit]
# new_df=new_df[new_df['Area'] > upper_limit_area]
percentile25 = new_df['PriceMC'].quantile(0.25)
percentile75 = new_df['PriceMC'].quantile(0.75)
area25=new_df['Area'].quantile(0.25)
area75=new_df['Area'].quantile(0.75)
iqr=percentile75 -percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

# df[df['PriceMC'] > upper_limit]
# df[df['PriceMC'] < lower_limit]
# df[df['Area'] > upper_limit]
# df[df['Area'] < lower_limit]
newdf =new_df[new_df['PriceMC'] < upper_limit]
newdf=newdf[newdf['PriceMC'] > lower_limit]
area25=new_df['Area'].quantile(0.25)
area75=new_df['Area'].quantile(0.75)
iqrArea=area75 -area25
newdf
upper_limit_area = area75 + 1.5 * iqrArea
lower_limit_area = area25 - 1.5 * iqrArea
# df[df['PriceMC'] > upper_limit]
# df[df['PriceMC'] < lower_limit]

newdf =new_df[new_df['Area'] < upper_limit_area]
# new_df =df[df['Area'] < upper_limit_area]

newdf=newdf[new_df['Area'] > lower_limit_area]
# new_df=new_df[new_df['Area'] > upper_limit_area]
x=newdf.drop('PriceMC',axis=1)

y=newdf['PriceMC']
xtrain , xtest ,ytrain ,ytest = train_test_split( x , y , test_size=20)
# رگرسیون خطی
liModel = LinearRegression()
liModel .fit(xtrain , ytrain)

# درخت تصمیم
treeModel=DecisionTreeRegressor().fit(xtrain,ytrain)

# بردار پشتیبان
svmModel=SVR(kernel='rbf',C=1e3,gamma=0.1)
svmModel.fit(xtrain,ytrain)



col1, col2, col3 ,col4= st.columns(4)
def user_input():
    room = st.sidebar.slider('Room',0 , 4 , 2)
    address = st.sidebar.slider('Address',1 , 22 , 5)
    area = st.sidebar.slider('Area',25 , 200 , 100)
    elevator = st.sidebar.checkbox('Elevator')
    warehouse = st.sidebar.checkbox('Warehouse')
    parking = st.sidebar.checkbox('Parking')
    d = {
        'Area': [area],
        'Room': [room],
        'Parking': [parking],
        'Warehouse': [warehouse],
        'Elevator': [elevator],
        'Address': [address]
    }

    # Creating a DataFrame
    forPridict = pd.DataFrame(d)
    return forPridict

forPridict=user_input()
predictByLiModel=liModel.predict(forPridict)
predictByTreeModel=treeModel.predict(forPridict)
predictBySvmModel=svmModel.predict(forPridict)
# predictByLiModel=liModel.predict(forPridict)
with col1:
    # st.subheader("predict by LR ")
    prediction = liModel.predict(xtest)
    score = r2_score(ytest, prediction)
    st.metric(label="predict by LR", value=predictByLiModel)
    st.success(format(round(score, 2) * 100) + "%")
with col2:
    # st.subheader("predict by tree ")
    prediction = treeModel.predict(xtest)
    score = r2_score(ytest, prediction)
    st.metric(label="predict by tree", value=predictByTreeModel)
    st.success(format(round(score, 2) * 100) + "%")
with col3:
    # st.subheader("predict by SVR ")
    prediction = svmModel.predict(xtest)
    score = r2_score(ytest, prediction)
    st.metric(label="predict by SVR" ,value=predictBySvmModel)
    st.success(format(round(score, 2) * 100) + "%")


    def _is_distributed_dataset(ds):
        #   return isinstance(ds, input_lib.DistributedDatasetInterface)
        return isinstance(ds, input_lib.DistributedDatasetSpec)
with col4:

   # st.subheader("predict by Neural Network ")
   dataset = newdf.values
   dataset=np.asarray(dataset).astype('float32')
   X=dataset[:,0:6]
   Y=dataset[:,6]

   from sklearn.preprocessing import MinMaxScaler
   min_max_scaler=MinMaxScaler()
   X_scale = min_max_scaler.fit_transform(X)

   Xtrain , Xtest ,Ytrain,Ytest = train_test_split(X_scale , Y , test_size=0.2)
   model = keras.Sequential([
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(1)
    ])
   model.compile(optimizer='adam', loss='mean_squared_error')
   model.fit(Xtrain, Ytrain, epochs=100, batch_size=32, validation_split=0.1)
   forestpridict=model.predict(Xtest)
   score = r2_score(Ytest,forestpridict)
   st.success(format(round(score, 2) * 100) + "%")
   majhol=np.asarray(forPridict).astype('float32')
   forestpridict=model.predict(majhol)
   st.metric(label="predict by neural network", value=forestpridict)