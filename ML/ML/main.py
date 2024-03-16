import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.metrics import r2_score
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import linregress
plt.style.use('fivethirtyeight')
import os
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(page_title="predictIran" , page_icon=":bar_chart:" ,layout="wide")
st.title(" :bar_chart: Crypto Price Prediction")
# st.markdown()
fl = st.sidebar.file_uploader(":file_folder: Upload a file",type=(["csv","txt"]))
if fl is not None:
    filename=fl.name
    st.write(filename)
    df = pd.read_csv(filename ,encoding="ISO-8859_1")
else:
    os.chdir(r"C:\Users\MRSZAMINI\Desktop\ML")
    df=pd.read_csv("Saipa.csv",encoding="ISO-8859-1")

df["Date"]=pd.to_datetime(df["Date"])






n = st.sidebar.text_input('How many days you wanna predict? ', 5)

if df is not None:
    st.header(filename+ ' Close Price\n')
else:
    st.header("Please")
st.line_chart(df['Close'])

st.header('Price Movement')
df2=df
df2['% Change']=df['Close'] / df['Close'].shift(1) - 1
st.write(df2[['Close','% Change','Low','High','Volume']])
# st.header(company + ' Price\n')
# st.line_chart(df['Close'])Z
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

tab1,tab2 ,tab3 ,tab4 = st.tabs(["SVR Regression" ,"Linear Regression ", "Decision Tree Regression" , "Neural Network"])

#########################################################SVR##########################
mysvr=SVR(kernel='rbf',C=1e3,gamma=0.1)
mysvr.fit(xtrain,ytrain)
svmpred=mysvr.predict(xtest)
valid=y[training_Y_len:]

valid['PredictionByModel']=svmpred

score = r2_score(valid['Prediction'], valid["PredictionByModel"])
print("The accuracy of our model is {}%".format(round(score, 2) *100))
with tab1:
    st.header("The accuracy of our model is ")
    st.success(score)
    forecast_range = forecast
    pred_df = pd.DataFrame()
    pred_df['Date'] = pd.date_range(start=df.Date.iloc[-1], periods=forecast_range + 1, closed='right')
    pred_df = pred_df.set_index(pd.DatetimeIndex(pred_df['Date'].values))
    pred_df['Close'] = mysvr.predict(x_forecast)
    valid = df[['Close']]


    fig,ax =plt.figure(figsize=(16, 8))
    ax.title('Price Predictor')
    ax.xlabel('Date')
    ax.ylabel('Price')
    ax.plot(valid['Close'])
    ax.plot(pred_df['Close'])
    ax.legend(['Curent', 'Prediction'])
    ax.show()
    st.pyplot(fig)




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
    fig ,ax=plt.figure(figsize=(16, 8))
    plt.title('Price Predictor')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(valid['Close'])
    plt.plot(pred_df['Close'])
    plt.legend(['Curent', 'Prediction'])
    plt.show()
    st.pyplot(fig)




