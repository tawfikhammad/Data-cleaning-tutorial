import numpy as np
import pandas as pd

class IterativeImputer:
    def __init__(self, estimator, initial_strategy='mean', imputation_order='ascending', max_iter=10):
        self.estimator = estimator
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.max_iter = max_iter

    def fit_transform(self, X):
        df = X.copy()

        # Step 1: Handle initial strategy for missing values
        if self.initial_strategy == 'mean':
            df = df.fillna(df.mean())
        elif self.initial_strategy == 'median':
            df = df.fillna(df.median())
        elif self.initial_strategy == 'most_frequent':
            df = df.fillna(df.mode().iloc[0])

        # Step 2: Iterative imputation
        for _ in range(self.max_iter):
            # Create a list to store the index of rows with missing values
            missing_indices = []

            # Identify the rows with missing values and store their indices in missing_indices
            for i, row in df.iterrows():
                if row.isnull().any():
                    missing_indices.append(i)

            # Step 3: Determine the imputation order
            if self.imputation_order == 'ascending':
                missing_indices.sort()
            elif self.imputation_order == 'descending':
                missing_indices.sort(reverse=True)
            elif self.imputation_order == 'random':
                np.random.shuffle(missing_indices)

            # Step 4: Use the estimator to predict missing values
            for index in missing_indices:
                missing_row = df.loc[index]
                non_missing_rows = df.drop(index)[missing_row.notnull()]
                missing_columns = missing_row[missing_row.isnull()].index

                for column in missing_columns:
                    # Extract non-missing values for the current column
                    X = non_missing_rows.drop(column, axis=1).values
                    y = non_missing_rows[column].values

                    # Fit the estimator model
                    self.estimator.fit(X, y)

                    # Predict the missing value using the fitted model
                    missing_value = self.estimator.predict([missing_row.drop(column)])

                    # Update the missing value in the DataFrame
                    df.at[index, column] = missing_value

        return df

# Test the custom IterativeImputer class
from sklearn.linear_model import LinearRegression

data = {'Age': [25, 30, 35, np.nan, 45],
        'Income': [50000, 55000, np.nan, 65000, 70000],
        'Education': [16, np.nan, 18, 20, 16]}
df = pd.DataFrame(data)

# Create an instance of the custom IterativeImputer class and perform imputation
estimator = LinearRegression()
imputer = IterativeImputer(estimator=estimator, initial_strategy='mean', imputation_order='random')
df_imputed = imputer.fit_transform(df)

print(df_imputed)

