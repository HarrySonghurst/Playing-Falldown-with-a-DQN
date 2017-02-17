from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.initializations import normal
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

# def construct_model():
#     model = Sequential()
#     # 64 3*3 filters, valid border (no zero padding), without regularization for now.
#     # Will need to accept 64*48*4 volume of inputs. (4 deep = 4 last frames).
#     # note, tensorflow backend so input to model needs to be (samples, rows, cols, channels)
#     # As per DeepMinds atari DQN, first conv has 32 8*8 filters with stride 4.
#     # output of first conv will be (W - F + 2P)/S + 1 = (32,16,12)
#     model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(68, 52, 4), activation='relu'))
#     # second conv has 64 4*4 filters, stride 2, producing an activation volume of shape (64,7,5)
#     model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
#     # third conv has 64 3*3 filters, stride 1, producing a volume of shape (64, 5, 3)
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     # flatten the 5*3*64 activation volume to a single dimension of size 960
#     model.add(Flatten())
#     # ReLU fully connected 256
#     model.add(Dense(256, activation='relu'))
#     # finally connect to 3 output nodes, 0 = action 0, 1 = [0,1], 2 = [1, 0]
#     model.add(Dense(3, activation='linear'))
#     model.compile(loss='mse', optimizer='rmsprop')
#     return model


def construct_model():
    model = Sequential()
    # 64 3*3 filters, valid border (no zero padding), without regularization for now.
    # Will need to accept 64*48*4 volume of inputs. (4 deep = 4 last frames).
    # note, tensorflow backend so input to model needs to be (samples, rows, cols, channels)
    # As per DeepMinds atari DQN, first conv has 32 8*8 filters with stride 4.
    # output of first conv will be (W - F + 2P)/S + 1 = (32,16,12)
    model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(68,52,4), dim_ordering='tf'))
    model.add(Activation('relu'))
    # second conv has 64 4*4 filters, stride 2, producing an activation volume of shape (64,7,5)
    model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
    model.add(Activation('relu'))
    # third conv has 64 3*3 filters, stride 1, producing a volume of shape (64, 5, 3)
    model.add(Convolution2D(64, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))
    # flatten the 5*3*64 activation volume to a single dimension of size 960
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    # finally connect to 3 output nodes, 0 = [0, 0], 1 = [0,1], 2 = [1, 0]. Note: default activation is linear, f(x)=x.
    model.add(Dense(3))

    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    print("Model initialised.")
    return model