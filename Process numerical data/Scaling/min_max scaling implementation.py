import numpy as np

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

# Example usage
data = np.array([2, 5, 10, 8, 4, 13, 50, 3, 24])
scaled_data = min_max_scaling(data)

print(scaled_data)
