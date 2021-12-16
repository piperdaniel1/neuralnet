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
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

crypto_currency = 'BTC'
against_currency = 'USD'

model = gen_model()
offset = dt.timedelta(days=60)
test_start = dt.datetime(2020, 1, 1) - offset
test_end = test_start + dt.timedelta(days=90)

data = web.DataReader(crypto_currency + "-" + against_currency, 'yahoo', test_start, test_end)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

full_data = scaled_data[0:60]
c = 0
while c < 30:
    test_data = full_data[-60:]
    test_data = np.reshape(test_data, (1, 60, 1))
    new_pred = model.predict(test_data)
    full_data = np.append(full_data, new_pred)
    c+=1

full_data = scaler.inverse_transform(full_data.reshape(-1, 1))

plt.plot(full_data, color='blue')
plt.plot(scaler.inverse_transform(scaled_data), color='red')
plt.show()