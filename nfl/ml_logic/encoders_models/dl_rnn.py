import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.optimizers import Adam

def ml_train(dl_pipe, X_train, y_train, X_test, y_test):
    # Reshape input data
    sequence_length = 20 # Dig into this
    X_train_rnn = X_train.reshape(X_train.shape[0], sequence_length, -1)
    X_test_rnn = X_test.reshape(X_test.shape[0], sequence_length, -1)

    # Define RNN model archi
    model_rnn = Sequential([
        SimpleRNN(64, input_shape = (sequence_length, X_train_rnn.shape[2]), return_sequences=True),
        SimpleRNN(32),
        Dense(16, activation = "relu"),
        Dense(5, activation = "softmax")
    ])

    # Compile model
    model_rnn.compile(optimizer = Adam(learning_rate = 0.01),
                        loss = "sparse_categorical_crossentropy",
                        metrics = ['precision'])

    # Train model
    history_rnn = model_rnn.fit(X_train_rnn, y_train,
                                epochs = 50,
                                batch_size = 32,
                                validation_split = .2,
                                verbose = 1)
