import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

history = pickle.load(open("scorehistory.p", "rb"))

frames = [x[2] for x in history]
epoch = [x[0] for x in history]
score = [y[1] for y in history]
avg = []

score_total = 0

for s in range(len(score)):
    score_total += score[s]
    if s != 0:
        avg.append(score_total/s)
    else:
        avg.append(score_total)

plt.plot(frames, score)

plt.show()
