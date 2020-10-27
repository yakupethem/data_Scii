# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 00:16:40 2020

@author: yakup
"""

import pandas as pd
import numpy as np



veriler=pd.read_csv("Churn_Modelling.csv")

x=veriler.iloc[:,3:13].values
y=veriler.iloc[:,13:].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
le2=LabelEncoder()

x[:,1]=le.fit_transform(x[:,1])

x[:,2]=le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer

ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")

x=ohe.fit_transform(x)
x=x[:,1:]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

import keras


from keras.layers import Dense
from keras.models import Sequential

classifier=Sequential()
classifier.add(Dense(6,init="uniform",activation="relu",input_dim=11))  

classifier.add(Dense(6,init = "uniform",activation = "relu") )
             
classifier.add(Dense(1,init = "uniform",activation = "sigmoid")) 
               
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(x_train,y_train,epochs=50)

y_tahmin=classifier.predict(x_test)
y_tahmin=(y_tahmin>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_tahmin)
"""
from keras.callbacks import EarlyStopping

ea=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
#classifier.fit(x_train,y_train,epochs=500,callbacks=[EA])
classifier.fit(x=x_train,y=y_train,verbose=1,validation_data=(x_test,y_test),epochs=700,callbacks=[ea])
"""