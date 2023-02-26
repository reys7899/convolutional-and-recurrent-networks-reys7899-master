"""
File Name: nn.py
The main code for the recurrent and convolutional networks assignment.
See README.md for details.
Author(s): Rey Sanayei
Version: 1.0 (11/21/2022)
"""
from typing import Tuple, List, Dict

import tensorflow


def create_toy_rnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    rnn_model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(units=64,
                                            return_sequences=True),input_shape=input_shape),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='linear')
    ])
    optimizer = tensorflow.keras.optimizers.Adam(0.01)
    rnn_model.compile(optimizer=optimizer, loss='mean_absolute_error',
                      metrics=['mean_absolute_error'])

    param_dict = {'batch_size': 10}
    res_tuple = (rnn_model, param_dict)

    return res_tuple


def create_mnist_cnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    cnn_model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tensorflow.keras.layers.MaxPool2D((2, 2)),

        tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPool2D((2, 2)),

        tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPool2D((2, 2)),

        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='softmax')
    ])
    optimizer = tensorflow.keras.optimizers.Adam(0.001)
    cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    param_dict = {'batch_size': 100}
    res_tuple = (cnn_model, param_dict)

    return res_tuple


def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    vocab_len = len(vocabulary)

    rnn_model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Embedding(input_dim=vocab_len + 1, output_dim=30),
        tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(64,
                                                                           return_sequences=True)),
        tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(32)),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dropout(0.5),

        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])
    optimizer = tensorflow.keras.optimizers.Adam(0.001)
    rnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    param_dict = {'batch_size': 32}
    res_tuple = (rnn_model, param_dict)

    return res_tuple


def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    vocab_len = len(vocabulary)
    cnn_model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Embedding(input_dim=vocab_len + 1, output_dim=10),

        tensorflow.keras.layers.Conv1D(16, 5, activation='relu'),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.GlobalMaxPooling1D(),
        tensorflow.keras.layers.Dense(8, activation='relu'),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])
    optimizer = tensorflow.keras.optimizers.Adam(0.001)
    cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    param_dict = {'batch_size': 64}
    res_tuple = (cnn_model, param_dict)

    return res_tuple
