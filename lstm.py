from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 


def lstm_model():
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(units=50))
    model.add(Dropout(0.25))

    model.add(Dense(units=1))
    model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['mse'])
    return model