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

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
x_olcekli=ss.fit_transform(xx)

ss2=StandardScaler()
y_olcekli=ss2.fit_transform(yy)

from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color="blue")
plt.show()

from sklearn.tree import DecisionTreeRegressor

tree=DecisionTreeRegressor()
tree.fit(xx,yy)

plt.scatter(xx,yy,color="red")
plt.plot(xx, tree.predict(xx), color="blue")
plt.show()

print(tree.predict([[11]]))
print(tree.predict([[6.6]]))

from sklearn.ensemble import RandomForestRegressor

forest=RandomForestRegressor(random_state=0,n_estimators=10)
forest.fit(xx,yy.ravel())

plt.scatter(xx,yy,color="red")
plt.plot(xx, forest.predict(xx), color="blue")
plt.show()
print(forest.predict([[6.6]]))

from sklearn.metrics import r2_score

print("-------------------")

print("random forest r2 değeri:",r2_score(yy,forest.predict(xx)))

print("decision tree r2 değeri:",r2_score(yy,tree.predict(xx)))

print("support vector r2 değeri:",r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("polynomial r2 değeri:",r2_score(yy,lin_reg4.predict(poly_reg4.fit_transform(xx))))

print("linear r2 değeri:",r2_score(yy,lin_reg.predict(xx)))







