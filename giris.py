# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")
print(veriler)

eksikveriler=pd.read_csv("eksikveriler.csv")
print(eksikveriler)


from sklearn.impute import SimpleImputer

impute=SimpleImputer(missing_values=np.nan,strategy="mean")

yas=eksikveriler.iloc[:,1:4].values
print(yas)

impute=impute.fit(yas)
yas=impute.transform(yas)
print(yas)