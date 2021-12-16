import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import full
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf

def gen_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    return model

# Train the model
checkpoint_path = "btc_model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

crypto_currency = 'BTC'
against_currency = 'USD'

model = gen_model()
total_profit = 0
money = 10000
c=0
while dt.datetime(2020, 1, 1) + dt.timedelta(days=7) * c + dt.timedelta(days=67) < dt.datetime.now():
    offset = dt.timedelta(days=7) * c
    test_start = dt.datetime(2020, 1, 1) + offset
    test_end = test_start + dt.timedelta(days=67)

    data = web.DataReader(crypto_currency + "-" + against_currency, 'yahoo', test_start, test_end)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    full_data = scaled_data[0:60]
    next_point = model.predict(np.reshape(full_data, (1, 60, 1)))

    pred_close = scaler.inverse_transform(next_point)
    actual_close = scaler.inverse_transform(scaled_data[-1:])
    current_close = scaler.inverse_transform(full_data[-1:])

    print("Prediction for period: " + str(test_start) + " to " + str(test_end))
    if current_close < pred_close:
        print("Price will go up!")
        profit_pct = (actual_close - current_close) / current_close
        print("Profit percentage if followed predicition: " + str(round(profit_pct[0][0] * 100, 2)) + "%")
        total_profit += round(profit_pct[0][0] * 100, 2)
    elif current_close > pred_close:
        print("Price will go down!")
        profit_pct = ((actual_close - current_close) / current_close) * -1
        print("Profit percentage if followed predicition: " + str(round(profit_pct[0][0] * 100, 2)) + "%")
        total_profit += round(profit_pct[0][0] * 100, 2)
    else:
        print("Price will stay the same!")
        print("Profit percentage if followed predicition: 0%")

    money *= (1 + profit_pct[0][0])
    print("Current close: $" + str(current_close))
    print("Predicted one week close: $" + str(scaler.inverse_transform(next_point)))
    print("Actual one week close: $" + str(scaler.inverse_transform(scaled_data[-1:])))
    print("Difference %: " + str(((scaler.inverse_transform(next_point) - scaler.inverse_transform(scaled_data[-1:])) / scaler.inverse_transform(scaled_data[-1:])) * 100))
    print("Total profit: $" + str(total_profit))
    print("Money: $" + str(money))
    print("\n")
    #plt.plot(scaler.inverse_transform(scaled_data), color='blue')
    #plt.plot(scaler.inverse_transform(full_data), color='green')
    #plt.scatter(67, scaler.inverse_transform(next_point), color='red')
    #plt.show()
    c+=1