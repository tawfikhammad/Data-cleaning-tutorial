# Import Libraries
import pandas as pd
import numpy as np
from statsmodels.imputation import mice
import statsmodels as sm


data = {'CustomerID': [1, 2, 3, 4, 5],
        'Income': [50000, 55000, np.nan, 65000, 70000],
        'Gender': ['m', np.nan, 'f', 'f', 'm'],
        'LastPurchaseAmount': [120.50, 85.30, np.nan, 200.00, np.nan]}

df = pd.DataFrame(data)

gender_dummies = pd.get_dummies(df.Gender, prefix='gender')
df = pd.concat([df, gender_dummies], axis=1)
df.gender_f = np.where(df.Gender.isna(), float('NaN'), df.gender_f)
df.gender_m = np.where(df.Gender.isna(), float('NaN'), df.gender_m)
df = df.drop(['Gender'], axis=1)



MI_data_df = mice.MICEData(df)
fit = mice.MICE(model_formula='LastPurchaseAmount~Income+gender_F+gender_M',
model_class=sm.api.OLS, data=MI_data_df)
MI_summ = fit.fit().summary()
print(MI_summ)