import numpy as np

three_D = np.zeros((2, 3, 3))
print(three_D)

print("Testing 3D Array:")
three_D = np.append(three_D, np.full((2, 3, 1), 1), axis=2)
print(three_D)