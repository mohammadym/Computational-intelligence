# Q1.2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1.2_graded
# Do not change the above line.

# This cell is for your codes.

templates = [[1, -1, 1, -1, 1, -1], [1, 1, 1, -1, -1, -1]]

weights = np.ndarray((6, 6), buffer=np.full(6*6, 0), dtype=float)
for item in templates: 
  for i in range(6):
    for j in range(i):
      weights[i, j] += item[i] * item[j]
      weights[j, i] = weights[i, j] 
      

np.fill_diagonal(weights, 0)
print(weights)
print(np.sign(np.dot(templates[0], weights)))
print(np.sign(np.dot(templates[1], weights)))
energy_of_network = np.ndarray((len(templates), 1), 
                               buffer=np.full(len(templates), 0), dtype=float)

for ind, item in enumerate(templates):
  for i in range(6):
    for j in range(6):
      energy_of_network[ind] = weights[i][j] * item[i] * item[j]
energy_of_network /= 2

print(energy_of_network)

