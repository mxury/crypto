import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


class dataset_generator():

    def __init__(self, df_train, df_val, seq_length, target_column=['close']):
        self.df_train = df_train
        self.df_val = df_val

        self.seq_length = seq_length

        self.train_mean = self.df_train.mean()
        self.train_std = self.df_train.mean()

        self.target_column = target_column

    def normalise(self, df):
        return (df - self.train_mean) / self.train_std

    def create_dataset(self, df):
        data = self.normalise(df)

        input_data = data.iloc[:-self.seq_length]
        target = data[self.target_column].iloc[self.seq_length:]

        dataset = timeseries_dataset_from_array(input_data, target, sequence_length=self.seq_length, shuffle=False)

        return dataset

    def train(self):
        return self.create_dataset(self.df_train)

    def val(self):
        return self.create_dataset(self.df_val)

    def return_numpy(self, df):
        inputs = np.array([])
        targets = np.array([])
        for batch in df:
            inputs = np.append(inputs, np.array(batch[0]))
            targets = np.append(targets, np.array(batch[1]))

        return targets

    def plot_target(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(15, 10))

        targets = self.return_numpy(self.train())

        ax.plot(targets)

