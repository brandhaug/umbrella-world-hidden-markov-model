import pip
print(pip.__version__)
print("senor")

import numpy as np

dynamic_model = np.array([
    [0.7, 0.3],
    [0.3, 0.7]
])

transition_model = np.array([
    [0.9, 0.3],
    [0.3, 0.9]
])

