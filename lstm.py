from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from lstm_data_gen import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def lstm_model(num_features, sequence_length=100):
    model = Sequential()

    model.add(Input(shape=(sequence_length, num_features)))

    model.add(LSTM(units=80, recurrent_dropout=0.25, return_sequences=True))
    model.add(LSTM(units=50, recurrent_dropout=0.25))

    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['mae'])
    return model


def cv_score(cv_folds, df, model_params):
    """
    Evaluates my LSTM model given a split.
    :param cv_folds: Cross validation split iterator
    :param df: Dataframe containing the data
    :return:
    """
    scores = list()
    cv_history = list()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00001,  # minimium amount of change to count as an improvement
        patience=5,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
            filepath='saved_models/model_{epoch}',
            save_freq='epoch')

    for train, val in cv_folds.split(df):
        df_train = df.iloc[train]
        df_val = df.iloc[val]
        dataset = dataset_generator(df_train, df_val, seq_length=model_params['seq_length'])

        dataset_train = dataset.train()
        dataset_val = dataset.val()

        print(f'Fitting fold number {len(cv_history)+1}')
        model = lstm_model()
        history = model.fit(dataset_train, epochs=model_params['epochs'], batch_size=model_params['batch_size'],
                            callbacks=[early_stopping],
                            validation_data=dataset_val,
                            )
        cv_history.append(history)

        scores.append(model.evaluate(dataset_val))
        model.save(f'./saved_models/model_{len(cv_history)}')

    return scores, cv_history, model

