# import necessary libraries
import numpy as np
import pandas as pd
 
# import the KNNimputer class
from sklearn.impute import KNNImputer
 
 
# create dataset for marks of a student
dict = {'Maths':[80, 90, np.nan, 95],
        'Chemistry': [60, 65, 56, np.nan],
        'Physics':[np.nan, 57, 80, 78],
       'Biology' : [78,83,67,np.nan]}
 
Before_imputation = pd.DataFrame(dict)

# before imputation
print(Before_imputation.isna().sum())
 
# create an object for KNNImputer
imputer = KNNImputer(n_neighbors=2)
After_imputation = imputer.fit_transform(Before_imputation)
df = pd.DataFrame(After_imputation, columns=['Maths', 'Chemistry', 'Physics','Biology'])

# after imputation
print(df.isna().sum())

