import numpy as np

filtered_data = np.arange(3*8*2).reshape((3, 8, 2))
print(filtered_data)
filtered_data = np.roll(filtered_data, -1, axis=1)
print(filtered_data)
filtered_data[0, 7] = [100, 101]
print(filtered_data)
