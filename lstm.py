from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from lstm_data_gen import *
from tensorflow.keras.callbacks import EarlyStopping


def lstm_model():
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.25))
    #
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.25))

    model.add(LSTM(units=50))
    model.add(Dropout(0.25))

    model.add(Dense(units=1))
    model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['mse'])
    return model


def cv_score(model, cv_folds, df, model_params):
    """
    Evaluates a Neural Network model given a split.
    :param model: Neural Network model
    :param cv_folds: Cross validation split iterator
    :param df: Dataframe containing the data
    :return:
    """
    scores = list()
    cv_history = list()
    early_stopping = EarlyStopping(
        min_delta=0.001,  # minimium amount of change to count as an improvement
        patience=10,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    model_checkpoint = [
        keras.callbacks.ModelCheckpoint(
            filepath='home/mikiu/Documents/git_projects/crypto/saved_models/model_{epoch}',
            save_freq='epoch')
    ]
    for train, val in cv_folds.split(df):
        df_train = df.iloc[train]
        df_val = df.iloc[val]
        dataset = dataset_generator(df_train, df_val, seq_length=model_params['seq_length'])

        dataset_train = dataset.train()
        print(f'Fitting fold number {len(cv_history)}')
        history = model.fit(dataset_train, epochs=model_params['epochs'], batch_size=model_params['batch_size'],
                            callbacks=[early_stopping, model_checkpoint],
                            )
        cv_history.append(history)

        dataset_val = dataset.val()
        scores.append(model.evaluate(dataset_val))

    return scores, cv_history

