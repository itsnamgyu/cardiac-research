import numpy as np

x = np.random.rand(10)
print("Original numpy:")
print(x)

np.save("example_data", x)
# This saves the data in x to file "example_file.npy"
# The .npy extension is automatically added, if the given
# filename doesn't already include .npy
print("Saved numpy data to example_data.npy")

print()

y = np.load("example_data.npy")
print("Loaded numpy data from example_data.npy")

print("Loaded numpy:")
print(y)
