# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 00:23:48 2020

@author: yakup
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veri=pd.read_csv("satislar.csv")
#veri. plot()

#print(veri)

aylar=veri[["Aylar"]]
satislar=veri[["Satislar"]]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(aylar, satislar, random_state=0, test_size=0.33)

"""
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(x_train)
X_test=ss.fit_transform(x_test)

Y_train=ss.fit_transform(y_train)
Y_test=ss.fit_transform(y_test)
"""

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()


plt.plot(x_train,y_train)
plt.plot(x_test, tahmin)













