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

def gen_model(model_day):
    checkpoint_path = model_day + "_day_model/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("Loading checkpoint:", checkpoint_path, checkpoint_dir)

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

crypto_currency = 'DOGE'
against_currency = 'USD'

total_profit = 0
money = 10000
c=0

early_date = dt.datetime.now() - dt.timedelta(days=65)
today = dt.datetime.now() 

data = web.DataReader(crypto_currency + "-" + against_currency, 'yahoo', early_date, today)
data = data.tail(61)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

full_data = scaled_data[-60:len(scaled_data)]
days = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh"]
predictions = []

for day in days:
    next_point = gen_model(day).predict(np.reshape(full_data, (1, 60, 1)))
    predictions.append(scaler.inverse_transform(next_point)[0][0])

    print(f"Predicted close on {day} day: $" + str(scaler.inverse_transform(next_point)))
    print("Predicted profit if you buy now and hold for seven days: " + str(round(((scaler.inverse_transform(next_point) - scaler.inverse_transform(scaled_data[-1:])) / scaler.inverse_transform(scaled_data[-1:]))[0][0] * 100, 2)) + "%")

print(data)

closes = scaler.inverse_transform(full_data)
blank = np.arange(len(closes), len(closes) + len(predictions))

plt.title(crypto_currency + "-" + against_currency + " Price Prediction")
plt.plot(closes, label='Price')
plt.plot(blank, predictions, label='Prediction')
plt.legend()
plt.show()