#!/usr/bin/env python
# coding: utf-8

# In[3]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
plt.style.use('fivethirtyeight')

# In[4]:

## Dashboard
st.title("  سامانه پیش بینی سرمایه برتر")




# img= Image.open('C:/Users/Hannane/Desktop/ML/img1.png')
# st.image(img,width=600)


# In[6]:


st.sidebar.header('INSERT DATA')
def data():
    n=st.sidebar.text_input('How many days you wanna predict? ',5)
    symbol=st.sidebar.selectbox('Select The Symbol : ',['Saipa','Khodro','Zamyad'])
    return n , symbol


# In[7]:


def get_data():
    if symbol=='Saipa':
        df=pd.read_csv('C:/Users/Hannane/Desktop/ML/Saipa.csv')
    elif symbol=='Khodro':
        df=pd.read_csv('C:/Users/Hannane/Desktop/ML/Khodro.csv')
    elif symbol=='Zamyad':
        df=pd.read_csv('C:/Users/Hannane/Desktop/ML/Zamyad.csv')

        
    df=df.set_index(pd.DatetimeIndex(df['Date'].values)) 
    return df


# In[8]:


def get_company_name(symbol):
    if symbol=='Saipa':
        return 'Saipa'
    elif symbol== 'Khodro':
        return 'KHODRO'
    elif symbol== 'Zamyad':
        return 'Zamyad'
    else :
        return  'NONE'


# In[9]:

n , symbol = data()
df=get_data()
company=get_company_name(symbol)
st.header(company + ' Close Price\n')
st.line_chart(df['CLOSE'])
# def plot_raw_date():
#     fig=go.Figure()
#     fig.add_trace(go.Scatter(x=df['Date'], y=data['OPENINT'],name='stock_open'))
#     fig.add_trace(go.Scatter(x=df['Date'], y=data['CLOSE'],name='stock_close'))
#     fig.layout.update(title_text="Time Series Date" , xaxis_rangeslider_visible=True)
#     st.pyplot_chart(fig)
#
# plot_raw_date()
st.header(company + 'Volume\n')
st.line_chart(df['VOL'])
st.header('Stock Datas')
st.write(df.describe())

tab1,tab2 = st.tabs(["SVM Algoritm" ,"Regre"])
# In[11]:


df=df[['CLOSE']]
forecast=int(n)
# داده های ما به تعداد ارگومان جا به جا می شود
df['Prediction']=df[['CLOSE']].shift(-forecast)
#  ستون را حذف میکند
x= np.array(df.drop(['Prediction'],1))

x= x[:-forecast]

y= np.array(df['Prediction'])
y=y[:-forecast]


# In[12]:


xtrain , xtest , ytrain , ytest=train_test_split(x,y,test_size=0.2,random_state=0)
mysvr=SVR(kernel='rbf',C=1000,gamma=0.1)
mysvr.fit(xtrain,ytrain)
svmconf=mysvr.score(xtest,ytest)
with tab1:
    st.header('SVM Accuracy')
    st.success(svmconf)
    x_forecast=np.array(df.drop(['Prediction'],1))[-forecast:]
    svmpred=mysvr.predict(x_forecast)
    st.header('SVM Prediction')
    st.success(svmpred)

# In[13]:





# In[14]:


lr=LinearRegression()
lr.fit(xtrain,ytrain)
lrconf=lr.score(xtest,ytest)
with tab2:

    st.header('LR Accuracy')
    st.success(lrconf)
    lrpred=lr.predict(x_forecast)
    st.header('LR Prediction')
    st.success(lrpred)
    # rmse =np.sqrt(np.mean(((lrpred- ytest)**2)))
    # st.write(rmse)

# In[15]:





# In[ ]:




