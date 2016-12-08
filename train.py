from game import falldown
import Q_network
import random
import numpy as np

# hyperparameters

epochs = 1000
discount_factor = 0.9
epsilon = 1.0
gamma = 0.975
batch_size = 40
memory_size = 80
memory = []
index_to_replace = 0
#           stay   left    right
actions = [[0, 0], [0, 1], [1, 0]]

model = Q_network.get_model()

for i in range(epochs):

    environment = falldown.Environment()
    state = environment.tick()
    running_status = True

    while running_status:
        # We're in state S, run the Q network forwards on S to obtain
        # Q values of all possible actions, which then allows us to make
        # an epsilon greedy move.
        Q_value = model.predict(state.reshape(1,64), batch_size=1)
        if np.random.random() < epsilon:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(Q_value)

        new_state, reward, running_status = environment.tick(actions[action])

        # Experience replay storage, if less than buffer size, store a tuple of (S, A, St+1, Rt+1)
        if (len(memory) < memory_size):
            memory.append((state, action, new_state, reward))
        else:
            # surely this will mean that memory[0] sticks around for two iterations initially?
            # and that memory[79] and memory[80] never actually get replaced?
            if index_to_replace < memory_size-1:
                index_to_replace += 1
            else:
                index_to_replace = 0

            # replace the memories one by one, iteratively filling the it with new memories.
            memory[index_to_replace] = (state, action, new_state, reward)

            # now sample the memory for a mini-batch of memories to fit with
            mini_batch = random.sample(memory, batch_size)

            X_train = []
            y_train = []

            for this_memory in mini_batch:

                state, action, new_state, reward = this_memory
                old_Q_value = model.predict(state.reshape(1,64), batch_size=1)
                new_Q_value = model.predict(new_state.reshape(1,64), batch_size=1)
                # remember that maxQ essentially makes the network take into account
                # future expected rewards...
                maxQ = np.max(new_Q_value)
                target = np.zeros((1,4))
                target[:] = old_Q_value[:]
                if reward == -1:  # non-terminal state, keep playing
                    update = reward + discount_factor * maxQ
                else:
                    update = reward

                target[0][action] = update
                X_train.append(state.reshape(64,))
                y_train.append(target.reshape(4,))

                # turn the list of arrays into a single array with each row containing training example
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            print("Game {}".format(i))

            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)
            state = new_state
        if reward != -1:
            # terminal, so status bit = 0
            status = 0
    if epsilon > 0.1:
        epsilon -= 1/epochs
