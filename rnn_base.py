# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Importing the traning set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
#print(training_set.head())

training_set = training_set.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mc = MinMaxScaler()
training_set = mc.fit_transform(training_set)

#Getting the input the Output
X_train = training_set[0:1257] ##input stock price at t
y_train = training_set[1:1258] ##output stock price at t+1

  
#Reshaping
X_train = np.reshape(X_train,(1257, 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initializing the RNN
regressor = Sequential()

#Adding the input Layer into the LSTM
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None,1)))
#Dense Layer(Output)
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train,y_train,batch_size=32,epochs=200)

#Getting the real stoc price of 2017
#Importing the traning set
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
#print(training_set.head())

real_stock_price = test_set.iloc[:,1:2].values


#Getting the Prediction of Test data
inputs = real_stock_price
inputs = mc.transform(inputs)
#Reshaping
inputs = np.reshape(inputs,(20, 1, 1))

predicted_stock_price = regressor.predict(inputs)

predicted_stock_price = mc.inverse_transform(predicted_stock_price)

#Visualizing the Results
plt.plot(real_stock_price, color='red', label='Actual stock price')
plt.plot(predicted_stock_price, color='blue', label='Predicted stock price')
plt.title('Google Stock Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

#Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
print(rmse)







