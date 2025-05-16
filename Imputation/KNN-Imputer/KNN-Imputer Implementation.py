import numpy as np

class KNNImputer:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit_transform(self, X):
        # Make a copy of the input data to avoid modifying the original DataFrame
        df = X.copy()

        # Step 1: KNN imputation
        for column in df.columns:
            missing_indices = df[df[column].isnull()].index
            non_missing_indices = df[df[column].notnull()].index

            for index in missing_indices:
                # Calculate distances to other data points
                distances = []
                for i in non_missing_indices:
                    distance = np.linalg.norm(df.loc[index] - df.loc[i])
                    distances.append((i, distance))

                # Sort distances and get the nearest neighbors
                distances.sort(key=lambda x: x[1])
                nearest_neighbors = distances[:self.n_neighbors]

                # Calculate the imputed value as the mean of the nearest neighbors
                imputed_value = np.mean([df.loc[i[0], column] for i in nearest_neighbors])

                # Update the missing value in the DataFrame
                df.at[index, column] = imputed_value

        return df


# Example dataset with missing values
import pandas as pd

dict = np.array([
    [2.0, 1.0, 3.0, np.nan, 5.0],
    [np.nan, 4.0, 7.0, 2.0, 9.0],
    [6.0, np.nan, 3.0, 8.0, 1.0],
    [7.0, 8.0, 5.0, 4.0, np.nan],
])
data= pd.DataFrame(dict)


# Create an instance of the custom KNNImputer class and perform imputation
knn_imputer = KNNImputer(n_neighbors=3)
data_imputed = knn_imputer.fit_transform(data)

print(data_imputed)


