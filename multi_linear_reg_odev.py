# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("odev.csv")
print(veriler)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

veriler2=veriler.apply(LabelEncoder().fit_transform)
c=veriler2.iloc[:,:1]


ohe=OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu=pd.DataFrame(data=c,index=range(14),columns=(["o","r","s"]))
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test=train_test_split(sonveriler.iloc[:,:-1], sonveriler.iloc[:,-1:], random_state=0, test_size=0.33)



from sklearn.linear_model import LinearRegression

regres=LinearRegression()

regres.fit(x_train,y_train)

y_tahmin=regres.predict(x_test)


import statsmodels.api as sm

xx=np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)

x_liste=sonveriler.iloc[:,[0,1,2,3,4,5]].values
x_liste=np.array(x_liste,dtype=float)

model=sm.OLS(sonveriler.iloc[:,-1:],x_liste).fit()
print(model.summary())



x_test=x_test.iloc[:,1:]
x_train=x_train.iloc[:,1:]
regres.fit(x_train,y_train)
y_tahmin=regres.predict(x_test)






