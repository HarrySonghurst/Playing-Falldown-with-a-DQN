from game import falldown
import Q_network
from collections import deque
import random
import numpy as np
from keras.models import load_model
import os.path

# initialize the neural network
if os.path.isfile("model.h5"):
    model = load_model("model.h5")
else:
    model = Q_network.construct_model()

# hyperparameters
epochs = 1000
discount_factor = 0.9
epsilon = 0.1
gamma = 0.98
batch_size = 32
frames_before_training = 1000
memory_size = 10000
replay_memory = deque()
#           stay   left    right
actions = [[0, 0], [0, 1], [1, 0]]
t = 0
loss = 0

for epoch in range(epochs):

    # init a game
    environment = falldown.Environment()

    # obtain the first state by ticking
    frame_t, reward_t, terminal_status = environment.tick([0,0])

    # stack the initial 4 frames to produce the initial state and format it for Keras
    state_t = np.stack((frame_t, frame_t, frame_t, frame_t), axis=2)
    state_t = np.reshape(state_t, (1, state_t.shape[0], state_t.shape[1], state_t.shape[2]))

    # while environment is not terminal (game is lost)
    while terminal_status:

        # take an epsilon greedy action
        if np.random.random() < epsilon:
            action_t = actions[np.random.randint(0,3)]
        else:
            Q_values_t = model.predict(state_t, batch_size=1)
            action_t = actions[np.argmax(Q_values_t)]

        frame_t1, reward_t1, terminal_status_t1 = environment.tick(action_t)
        terminal_status = terminal_status_t1

        # using the observed returns of the new frame, construct a new state from the
        # frame itself and from the last 3 frames of state_t.
        frame_t1 = np.reshape(frame_t1, (1, frame_t1.shape[0], frame_t1.shape[1], 1))
        state_t1 = np.append(frame_t1, state_t[:, :, :, :3], axis=3)

        # append (s,a,s_t+1, r_t+1) tuple to the replay memory
        replay_memory.append((state_t, action_t, state_t1, reward_t1, terminal_status_t1))

        # pop the oldest item in memory if its over the specified memory_size
        if len(replay_memory) > memory_size:
            replay_memory.popleft()

        # only train after a certain amount of experience has been observed.
        if t > frames_before_training:
            print("Training")
            # sample a random minibatch of the replay memory to train on
            minibatch = random.sample(replay_memory, batch_size)

            # init X and y training arrays
            X_train = np.zeros((batch_size, state_t.shape[1], state_t.shape[2], state_t.shape[3]))
            y_train = np.zeros((batch_size, 3))

            for memory in range(batch_size):

                # unpack the memory in the minibatch
                state_t, action_t, state_t1, reward_t1, terminal_status_t1 = minibatch[memory]

                # make state_t the X for this training memory
                X_train[memory:memory+1] = state_t
                # corresponding target will be the prediction, with the value of index of
                # action taken changed.
                y_train[memory] = model.predict(state_t, batch_size=1)

                # if the game is over, the update = reward
                if not terminal_status_t1:
                    update = reward_t1 + gamma * np.max(model.predict(state_t1, batch_size=1))
                    y_train[memory][actions.index(action_t)] = update
                else:
                    y_train[memory][actions.index(action_t)] = reward_t1

            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)

        t += 1
        state_t = state_t1

        if t % 100 == 0:
            print("saving model")
            model.save("model.h5")

    print("Episode {} finished.".format(epoch))