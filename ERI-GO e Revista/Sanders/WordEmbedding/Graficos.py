import matplotlib.pyplot as plt
import numpy as np

acuracia = [78.56, 78.77, 79.15, 79.36]
precisao = [74.53, 75.17, 75.83, 75.82]
recall = [63.36, 63.68, 64.15, 64.89]
f1 = [67.25, 67.66, 68.26, 68.85]
dimensoes = [25, 50, 100, 200]

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, ax = plt.xticks()

ax.plot(dimensoes, acuracia)
ax.plot(dimensoes, precisao)
ax.plot(dimensoes, recall)
ax.plot(dimensoes, f1)
ax.set_title("'fivethirtyeight' style sheet")

plt.show()