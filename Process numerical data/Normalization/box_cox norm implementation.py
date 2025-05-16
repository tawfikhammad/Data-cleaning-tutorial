import numpy as np

def boxcox_transform(data, lmbda):
    if lmbda == 0:
        transformed_data = np.log(data)
    else:
        transformed_data = (np.power(data, lmbda) - 1) / lmbda
    return transformed_data

# Generate example data (positively skewed)
data = np.random.exponential(scale=2, size=100)

lmbda = 0.5

transformed_data = boxcox_transform(data, lmbda)
print("Transformed Data (scipy):", transformed_data)
