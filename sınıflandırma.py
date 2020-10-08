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

#çocuk yaştakiler veriyi bozuyor
x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test=train_test_split(x ,y, random_state=0, test_size=0.33)



from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(x_train)
X_test=ss.transform(x_test)

from sklearn.linear_model import LogisticRegression

log_reg= LogisticRegression(random_state=0)

log_reg.fit(X_train, y_train)

y_tahmin=log_reg.predict(X_test)

print(y_tahmin)
print(y_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_tahmin)
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train, y_train)

y_tahmin2=knn.predict(X_test)

cm=confusion_matrix(y_test,y_tahmin2)
print("knn: ")
print(cm)


from sklearn.svm import SVC

svc=SVC(kernel="rbf")

svc.fit(X_train, y_train)

y_tahmin3=svc.predict(X_test)

cm=confusion_matrix(y_test,y_tahmin3)
print("SVC: ")
print(cm)





