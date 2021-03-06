# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:18:18 2020

@author: yakup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

veriler=pd. read_csv("maaslar_yeni.csv")

x=veriler.iloc[:,2:5]
y=veriler.iloc[:,5:]

xx=x.values
yy=y.values


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(xx,yy)

model=sm.OLS(lin_reg.predict(xx),xx)
print(model.fit().summary())

from sklearn.preprocessing import PolynomialFeatures

poly_reg2=PolynomialFeatures(degree=2)
x_poly=poly_reg2.fit_transform(xx)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,yy)


poly_reg4=PolynomialFeatures(degree=4)
x_poly4=poly_reg4.fit_transform(xx)
lin_reg4=LinearRegression()
lin_reg4.fit(x_poly4,yy)


from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
x_olcekli=ss.fit_transform(xx)

ss2=StandardScaler()
y_olcekli=ss2.fit_transform(yy)

from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)


from sklearn.tree import DecisionTreeRegressor

tree=DecisionTreeRegressor()
tree.fit(xx,yy)


"""
print(tree.predict([[11]]))
print(tree.predict([[6.6]]))"""

from sklearn.ensemble import RandomForestRegressor

forest=RandomForestRegressor(random_state=0,n_estimators=10)
forest.fit(xx,yy.ravel())

#print(forest.predict([[6.6]]))

from sklearn.metrics import r2_score

print("-------------------")

print("random forest r2 değeri:",r2_score(yy,forest.predict(xx)))

print("decision tree r2 değeri:",r2_score(yy,tree.predict(xx)))

print("support vector r2 değeri:",r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("polynomial r2 değeri:",r2_score(yy,lin_reg4.predict(poly_reg4.fit_transform(xx))))

print("linear r2 değeri:",r2_score(yy,lin_reg.predict(xx)))

print(veriler.corr())





