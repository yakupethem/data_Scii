# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_excel("iris.xls")
print(veriler)

#çocuk yaştakiler veriyi bozuyor
x=veriler.iloc[:,:4].values
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

"""
import math
n=math.sqrt(len(veriler.index))/2
"""

knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

p=[{"n_neighbors":[1,2,3,4,5]}]

gs=GridSearchCV(estimator=knn,param_grid=p,
                scoring="accuracy",cv=10,n_jobs=10)

sonuc=gs.fit(x_train,y_train)
eniyisonuc=sonuc.best_score_
eniyiparametreler=sonuc.best_params_
print("en iyi sonuç ",eniyisonuc)  
print("en iyi parametre ",eniyiparametreler)


y_tahmin2=knn.predict(X_test)

cm=confusion_matrix(y_test,y_tahmin2)
print("knn: ")
print(cm)

"""
from sklearn.svm import SVC

svc=SVC(kernel="rbf")

svc.fit(X_train, y_train)

y_tahmin3=svc.predict(X_test)

cm=confusion_matrix(y_test,y_tahmin3)
print("SVC: ")
print(cm)

from sklearn.naive_bayes import GaussianNB

gaus=GaussianNB()
gaus.fit(X_train,y_train)

y_tahmin4=gaus.predict(X_test)

cm=confusion_matrix(y_test,y_tahmin4)
print("Gaussian: ")
print(cm)


from sklearn.tree import DecisionTreeClassifier,plot_tree

dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_tahmin5=dtc.predict(X_test)

cm=confusion_matrix(y_test,y_tahmin5)
print("DTC Classification: ")
print(cm)
plot_tree(dtc)


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(X_train,y_train)
y_tahmin6=rfc.predict(X_test)

cm=confusion_matrix(y_test,y_tahmin6)
print("random forest Classification: ")
print(cm)


#ROC

y_proba=rfc.predict_proba(X_test)
print(y_test,"----",y_proba[:,0])


from sklearn.metrics import roc_curve,auc

fpr,tpr,trshold=roc_curve(y_test,y_proba[:,0],pos_label="e")
print("fpr: ",fpr)
print("tpr: ",tpr)


"""


























