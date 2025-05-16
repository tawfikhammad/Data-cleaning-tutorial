# Import libraries
import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

# Create a simple dataset
np.random.seed(0)
data = {'Age': [25, 30, 35, np.nan, 45],
        'Income': [50000, 55000, np.nan, 65000, 70000],
        'Education': [16, np.nan, 18, 20, 16]}
df = pd.DataFrame(data)

# Apply Predictive Mean Matching
imputer = IterativeImputer(estimator=LinearRegression(), initial_strategy='mean', imputation_order='random')
df_imputed = imputer.fit_transform(df)


df_imputed = pd.DataFrame(df_imputed, columns=['Age', 'Income', 'Education'])