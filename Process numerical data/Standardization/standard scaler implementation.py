import numpy as np

def standardize(X):
  
  # Calculate the mean and std of each feature.
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)

  # Subtract the mean and divide by the standard deviation for each feature.
  standardized = (X - mean) / std

  return standardized

data = np.random.rand(100, 3)  # A 2D array with 100 samples and 3 features

standardized_data = standardize(data)
print(standardized_data)
