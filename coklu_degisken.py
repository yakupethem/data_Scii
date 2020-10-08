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


ulke=veriler.iloc[:,0:1].values
print(ulke)

c=veriler.iloc[:,-1:].values
print(c)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le=LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)

ohe=OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=(["fr","us","tr"]))

sonuc2=pd.DataFrame(data=yas,index=range(22),columns=(["boy","kilo","yas"]))

cinsiyet=veriler.iloc[:,-1].values
sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=(["cinsiyet"]))

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test=train_test_split(s, sonuc3, random_state=0, test_size=0.33)



from sklearn.linear_model import LinearRegression

regres=LinearRegression()

regres.fit(x_train,y_train)

y_tahmin=regres.predict(x_test)

boy=s2.iloc[:,3:4].values
print(boy)

sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]

veri=pd.concat([sol,sag],axis=1) 

x_train, x_test,y_train, y_test=train_test_split(veri, boy, random_state=0, test_size=0.33)

r2=LinearRegression()

r2.fit(x_train,y_train)

y_tahmin=r2.predict(x_test)


import statsmodels.api as sm

xx=np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)

x_liste=veri.iloc[:,[0,1,2,3,4,5]].values
x_liste=np.array(x_liste,dtype=float)

model=sm.OLS(boy,x_liste).fit()
print(model.summary())

x_liste=veri.iloc[:,[0,1,2,3,5]].values
x_liste=np.array(x_liste,dtype=float)

model=sm.OLS(boy,x_liste).fit()
print(model.summary())

x_liste=veri.iloc[:,[0,1,2,3]].values
x_liste=np.array(x_liste,dtype=float)

model=sm.OLS(boy,x_liste).fit()
print(model.summary())









