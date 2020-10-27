# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:18:18 2020

@author: yakup
"""

import pandas as pd
import numpy as np

veriler=pd. read_csv("Social_Network_Ads.csv")

x=veriler.iloc[:,2:3].values
y=veriler.iloc[:,4].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.33)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.svm import SVC

classification=SVC(kernel='rbf',random_state=0)
classification.fit(x_train,y_train)

y_tahmin=classification.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_tahmin)
print("cm: ",cm)


#k_fold katlamalı çapraz doğrulama
from sklearn.model_selection import cross_val_score
basari=cross_val_score(estimator=classification,X=x_train,y=y_train,cv=2)

print(basari.mean())  #accuracy 1'e yakın iyi
print(basari.std())   #0'a yakın iyi
print(classification.score(x_test,y_test)) 









