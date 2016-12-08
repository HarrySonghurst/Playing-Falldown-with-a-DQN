from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D


def get_model():

    model = Sequential()
    # 64 3*3 filters, valid border (no zero padding), without regularization for now. W_regularizer=l2(l=0.01)
    model.add(Convolution2D(32, 3, 3, input_shape=(64, 48), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')

    return model
