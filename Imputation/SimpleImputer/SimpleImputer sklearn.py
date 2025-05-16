import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# create dataset for marks of a student
dict = {'Maths':[80, 90, np.nan, 95],
        'Chemistry': [60, 65, 56, np.nan],
        'Physics':[np.nan, 57, 80, 78],
       'Biology' : [78,83,67,np.nan]}
 

# creating a data frame from the list
df = pd.DataFrame(dict)


imputer = SimpleImputer(strategy='mean')
imputed = imputer.fit_transform(df)


df_imputed = pd.DataFrame(imputed, columns=df.columns)
print(df_imputed)
