# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:15:50 2020

@author: yakup
"""

import numpy as np
import pandas as pd

veriler=pd.read_csv("Wine.csv")

x=veriler.iloc[:,:13].values
y=veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from sklearn.decomposition import PCA
pca=PCA(n_components=2)

x_train2=pca.fit_transform(x_train)
x_test2=pca.transform(x_test)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(x_train,y_train)

lr2=LogisticRegression(random_state=0)
lr2.fit(x_train2,y_train)

y_tahmin=lr.predict(x_test)
y_tahmin2=lr2.predict(x_test2)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_tahmin)
cm2=confusion_matrix(y_test,y_tahmin2)
cm3=confusion_matrix(y_tahmin,y_tahmin2)

print("gerçek-PCA'sız: ",cm)
print("gerçek-PCA'lı: ",cm2)
print("PCA'sız-PCA'lı: ",cm3)
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       