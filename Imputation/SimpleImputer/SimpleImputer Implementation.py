import numpy as np
import pandas as pd

class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_value = None

    def fit(self, X):
        if self.strategy == 'mean':
            self.fill_value = X.mean()
        elif self.strategy == 'median':
            self.fill_value = X.median()
        elif self.strategy == 'most_frequent':
            self.fill_value = X.mode().iloc[0]
        else:
            raise ValueError("Invalid strategy. Supported strategies: 'mean', 'median', 'most_frequent'")
        
        return self

    def transform(self, X):
        if self.fill_value is None:
            raise ValueError("Imputer has not been fitted.")
        
        X_transformed = X.copy()
        missing_mask = X.isnull()
        for col in X.columns:
            col_missing = missing_mask[col]
            X_transformed.loc[col_missing, col] = self.fill_value[col]

        return X_transformed

# Example usage for handling categorical data:
# Test the custom SimpleImputer class
data = {'Age': [25, 30, 35, np.nan, 45],
        'Income': [50000, 55000, np.nan, 65000, 70000],
        'Education': [16, np.nan, 18, 20, 16],
        'color': ['red', np.nan, 'blue', 'green', 'red']}

df = pd.DataFrame(data)

imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(df)  # Fit the imputer on the data

imputed_data = imputer.transform(df) 

print(imputed_data)

