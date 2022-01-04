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
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("Loading checkpoint:", checkpoint_path, checkpoint_dir)

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(90, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=14))

    model.compile(optimizer='adam', loss='mean_squared_error')
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    return model

crypto_currency = 'BTC'
against_currency = 'USD'

total_profit = 0
money = 10000
c=0
distance = 0

early_date = dt.datetime.now() - dt.timedelta(days=95)#- dt.timedelta(days=14 * distance)
today = dt.datetime.now()#- dt.timedelta(days=14 * distance)
future_data = today + dt.timedelta(days=14)

data = web.DataReader(crypto_currency + "-" + against_currency, 'yahoo', early_date, today)
#real_data = web.DataReader(crypto_currency + "-" + against_currency, 'yahoo', today, future_data)['Close']
#real_data = list(real_data)

data = data.tail(91)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

full_data = scaled_data[-90:len(scaled_data)]
predictions = []

predictions = gen_model("training_2").predict(np.reshape(full_data, (1, 90, 1)))
predictions = scaler.inverse_transform(predictions)
new_pred = []
closes = scaler.inverse_transform(full_data)

new_pred.append(closes[-1])
for pred in predictions[0]:
    new_pred.append(pred)

#print("Data", real_data)
#real_data.append(real_data[-1])
print("Prediction:", predictions)

blank = np.arange(len(closes)-1, len(closes) + len(new_pred) - 1)

plt.title(crypto_currency + "-" + against_currency + " Price Prediction")
plt.plot(closes, label='Price')
plt.plot(blank, new_pred, label='Prediction')
#plt.plot(blank, real_data, label='Actual')
plt.legend()
plt.show()