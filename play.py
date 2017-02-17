from game import falldown
import Q_network
import numpy as np
from keras.optimizers import Adam
import os.path

# initialize the neural network
if os.path.isfile("model.h5"):
    model = Q_network.construct_model()
    model.load_weights("model.h5")
    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
else:
    raise Exception('Import error, model.h5 not found.')

#           stay   left    right
actions = [[0, 0], [0, 1], [1, 0]]

while True:

    # init a game
    environment = falldown.Environment(novid=False)

    # obtain the first state by ticking
    frame_t, reward_t, terminal_status = environment.tick([0, 0])

    # stack the initial 4 frames to produce the initial state and format it for Keras
    state_t = np.stack((frame_t, frame_t, frame_t, frame_t), axis=2)
    state_t = np.reshape(state_t, (1, state_t.shape[0], state_t.shape[1], state_t.shape[2]))

    # while environment is not terminal (game is lost)
    while terminal_status:

        Q_values_t = model.predict(state_t, batch_size=1)
        action_t = actions[np.argmax(Q_values_t)]

        frame_t1, reward_t1, terminal_status_t1 = environment.tick(action_t)
        terminal_status = terminal_status_t1

        # using the observed returns of the new frame, construct a new state from the
        # frame itself and from the last 3 frames of state_t (:3).
        frame_t1 = np.reshape(frame_t1, (1, frame_t1.shape[0], frame_t1.shape[1], 1))
        state_t1 = np.append(frame_t1, state_t[:, :, :, :3], axis=3)

        state_t = state_t1
