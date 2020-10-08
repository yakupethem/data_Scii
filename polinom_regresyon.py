# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:18:18 2020

@author: yakup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd. read_csv("maaslar.csv")

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

xx=x.values
yy=y.values


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(xx,yy)

from sklearn.preprocessing import PolynomialFeatures

poly_reg2=PolynomialFeatures(degree=2)
x_poly=poly_reg2.fit_transform(xx)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,yy)


poly_reg4=PolynomialFeatures(degree=4)
x_poly4=poly_reg4.fit_transform(xx)
lin_reg4=LinearRegression()
lin_reg4.fit(x_poly4,yy)



plt.scatter(xx,yy,color="red")
plt.plot(x,lin_reg.predict(xx),color="blue")
plt.show()

plt.scatter(xx,yy,color="red")
plt.plot(xx,lin_reg2.predict(poly_reg2.fit_transform(xx)),color="blue")
plt.show()

plt.scatter(xx,yy,color="red")
plt.plot(xx,lin_reg4.predict(poly_reg4.fit_transform(xx)),color="blue")
plt.show()








