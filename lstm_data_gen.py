import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


class dataset_generator():

    def __init__(self, X, cv, seq_length, sampling_rate,  target_column=['close']):
        self.X = X
        self.cv = cv

        self.seq_length = seq_length
        self.sampling_rate = sampling_rate
        self.delay = self.sampling_rate * self.seq_length

        self.target_column = target_column

        self.fold_data = list()

        self.gen()

    def gen(self):
        for train, val in self.cv.split(self.X):
            fold = LSTM_fold_data()
            fold.train = self.X.iloc[train]
            fold.val = self.X.iloc[val]
            fold.train_batch = self.create_lstm_tensor(fold.train_norm)
            fold.val_batch = self.create_lstm_tensor(fold.val_norm)
            self.fold_data.append(fold)

    def create_lstm_tensor(self, df):
        input_data = df.iloc[:-self.delay]
        target = df[self.target_column].iloc[self.delay:]
        return timeseries_dataset_from_array(input_data,
                                             target,
                                             sequence_length=self.seq_length,
                                             sampling_rate=self.sampling_rate,
                                             batch_size = 256,
                                             shuffle=True,
                                            )


class LSTM_fold_data():
    def __init__(self):
        self.train = None
        self.val = None
        self._train_mean = None
        self._train_std = None
        self.train_batch = None
        self.val_batch = None

    @property
    def train_std(self):
        if self.train is None:
            raise Exception("Train data has not been set! Can't compute std.")
        if self._train_std is None:
            self._train_std = self.train.std()
        return self._train_std

    @property
    def train_mean(self):
        if self.train is None:
            raise Exception("Training data has not been set! Can't compute the mean")
        if self._train_mean is None:
            self._train_mean = self.train.mean()
        return self._train_mean

    def normalise(self, df):
        return (df - self.train_mean) / self.train_std

    @property
    def train_norm(self):
        return self.normalise(self.train)

    @property
    def val_norm(self):
        return self.normalise(self.val)


