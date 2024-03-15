import numpy as np


a = np.array([True, False, True, True, True, False, False, True, False, True, True, False])

b = np.roll(a, 1)

equal = a == b

print(a)
print(b)
print(equal)

