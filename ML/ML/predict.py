import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import plotly.express as px
import streamlit as st
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.metrics import r2_score
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import linregress
plt.style.use('fivethirtyeight')
import yfinance as yf

st.set_page_config(page_title="predict" , page_icon=":bar_chart:" ,layout="wide")
st.title(" :bar_chart: Crypto Price Prediction")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.header('INSERT DATA')
col1 , col2 =st.columns((2))

def data():
    n=st.sidebar.text_input('How many days you wanna predict? ',5)
    symbol=st.sidebar.selectbox('Select The Symbol : ',['BTC-USD','AMZN','ETH-USD','TSLA'])
    # mydate = st.sidebar.selectbox('Select The start date : ' ,['2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01'] )
    mydate = st.sidebar.date_input('Start Data')
    return mydate , n , symbol



mydate , n , symbol = data()


def get_data():
    df = yf.Ticker(symbol)
    df = df.history(period='1d' , start= mydate )
    df['Date'] = df.index
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    return df



def get_company_name(symbol):
    if symbol=='BTC-USD':
        return 'BITCOIN'
    elif symbol== 'ETH-USD':
        return 'ETHEREUM'
    elif symbol== 'AMZN':
        return 'AMAZON'
    elif symbol=='TSLA':
        return 'TESLA'

    else :
        return  'NONE'



df=get_data()
company=get_company_name(symbol)
st.header('Price Movement')
df2=df
df2['% Change']=df['Close'] / df['Close'].shift(1) - 1
st.write(df2[['Close','% Change','Low','High','Volume']])
# st.header(company + ' Price\n')
# st.line_chart(df['Close'])Z
st.header(company + ' Volume\n')
st.line_chart(df['Volume'])
fig=px.line(df, x=df.index , y=df['Close'])
st.plotly_chart(fig)
# تنظیمات دیتا ست
forecast=int(n)
df['Prediction']=df[['Close']].shift(-forecast)
x=df.copy()
y=df.copy()
x=x[:-forecast]
x = x[['Close' , 'Volume']]
xarray = np.array(x)
x_forecast=xarray[-forecast:]
y = y[['Prediction']]
y= y[:-forecast]
yarray = np.array(y)

# داده های Train & Test
training_X_len=math.ceil(len(xarray)*0.8)
xtrain=xarray[0:training_X_len , :]
xtest=xarray[training_X_len : , :]
training_Y_len=math.ceil(len(yarray)*0.8)
ytrain=yarray[0:training_Y_len , :]
ytest=yarray[training_Y_len : , :]

tab2 ,tab3 ,tab4 = st.tabs(["Linear Regression ", "Decision Tree Regression" , "Neural Network"])

#########################################################SVR##########################
mysvr=SVR(kernel='rbf',C=1e3,gamma=0.1)
mysvr.fit(xtrain,ytrain)
svmpred=mysvr.predict(xtest)
valid=y[training_Y_len:]

valid['PredictionByModel']=svmpred

score = r2_score(valid['Prediction'], valid["PredictionByModel"])
print("The accuracy of our model is {}%".format(round(score, 2) *100))
# with tab1:
#     st.header("The accuracy of our model is ")
#     st.success(score)
#     forecast_range = forecast
#     pred_df = pd.DataFrame()
#     pred_df['Date'] = pd.date_range(start=df.Date.iloc[-1], periods=forecast_range + 1, closed='right')
#     pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['Date'].values))
#     pred_df['Close'] = mysvr.predict(x_forecast)
#     valid = df[['Close']]
#
#
#     plt.figure(figsize=(16, 8))
#     plt.title('Price Predictor')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.plot(valid['Close'])
#     plt.plot(pred_df['Close'])
#     plt.legend(['Curent', 'Prediction'])
#     plt.show()
#     st.pyplot()
#
#
#
#
lr=LinearRegression()
lr.fit(xtrain,ytrain)
with tab2:
    ytrain = yarray[0:training_Y_len, :]
    ytest = yarray[training_Y_len:, :]
    lrpred= lr.predict(xtest)
    valid = y[training_Y_len:]

    valid['PredictionByModel'] = lrpred
    score = r2_score(valid['Prediction'], valid["PredictionByModel"])
    print("The accuracy of our model is ".format(round(score, 2) * 100))
    st.header("The accuracy of our model is ")
    st.success(score)
    forecast_range = forecast
    pred_df = pd.DataFrame()
    pred_df['Date'] = pd.date_range(start=df.Date.iloc[-1], periods=forecast_range + 1, closed='right')
    pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['Date'].values))
    lrpred = lr.predict(x_forecast)
    pred_df['Close'] = lrpred
    valid = df[['Close']]
    plt.figure(figsize=(16, 8))
    plt.title('Price Predictor')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(valid['Close'])
    plt.plot(pred_df['Close'])
    plt.legend(['Curent', 'Prediction'])
    plt.show()
    st.pyplot()
with tab3:
    tree = DecisionTreeRegressor().fit(xtrain, ytrain)
    treered = tree.predict(xtest)
    valid = y[training_Y_len:]

    valid['PredictionByModel'] = treered
    forecast_range = forecast
    pred_test = pd.DataFrame()
    pred_test['Date'] = valid.index
    pred_test = pred_test.set_index(pd.DatetimeIndex(pred_test['Date'].values))
    pred_test['Close'] = valid[['PredictionByModel']].shift(forecast)
    pred_test['Value'] = valid[['Prediction']].shift(forecast)
    score = r2_score(valid['Prediction'], valid["PredictionByModel"])
    st.header("The accuracy of our model is ")
    st.success(score)
    forecast_range = forecast
    pred_df = pd.DataFrame()
    pred_df['Date'] = pd.date_range(start=df.Date.iloc[-1], periods=forecast_range + 1, closed='right')
    pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['Date'].values))
    treepred = tree.predict(x_forecast)
    pred_df['Close'] = tree.predict(x_forecast)
    valid = df[['Close']]
    plt.figure(figsize=(16, 8))
    plt.title('Price Predictor')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(valid['Close'])
    plt.plot(pred_df['Close'])
    plt.legend(['Curent', 'Prediction'])
    plt.show()
    st.pyplot()
with tab4:
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    training_data = scaled_data[0:training_data_len, :]
    xtrain = []
    ytrain = []
    n = 30
    for i in range(n, len(training_data)):
        xtrain.append(training_data[i - n:i, 0])
        ytrain.append(training_data[i, 0])
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xtrain, ytrain, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - n:, :]
    xtest = []
    ytest = dataset[training_data_len:, :]
    for i in range(n, len(test_data)):
        xtest.append(test_data[i - n:i, 0])

    xtest = np.array(xtest)
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

    prediction = model.predict(xtest)
    prediction = scaler.inverse_transform(prediction)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['prediction'] = prediction

    score = r2_score(ytest, valid['prediction'])
    st.header("The accuracy of our model is ")
    st.success(score)
    forecast_range = forecast
    new_df = df.filter(['Close'])
    # new_df[-n:].values
    pred_df = pd.DataFrame()
    pred_df['Date'] = pd.date_range(start=df.Date.iloc[-1], periods=forecast_range + 1, closed='right')

    pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['Date'].values))
    predictiondata = []
    for i in range(0, forecast_range):
        last_values = new_df[-n:].values
        last_values_scaled = scaler.transform(last_values)
        X_input = []
        X_input.append(last_values_scaled)
        X_input = np.array(X_input)
        X_test = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
        pred_value = model.predict(X_input)
        pred_value_unscaled = scaler.inverse_transform(pred_value)
        dfindex = pred_df.iloc[[i]].index
        new_df = new_df.append(pd.DataFrame({"Close": pred_value_unscaled[0, 0]}, index=dfindex))
        predictiondata.append(pred_value_unscaled[0, 0])
    pred_df['Close'] = predictiondata
    valid = df[['Close']]
    plt.figure(figsize=(16, 8))
    plt.title('Price Predictor')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(valid['Close'])
    plt.plot(pred_df['Close'])
    plt.legend(['Curent', 'Prediction'])
    plt.show()
    st.pyplot()