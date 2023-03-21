import numpy as np

three_D = np.zeros((2, 2, 2))
three_D[0, 0, 0] = 1
three_D[0, 0, 1] = 2

probabilities = three_D[0, 0, :]/np.sum(three_D[0, 0, :])

print(probabilities)
