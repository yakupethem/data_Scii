# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:26:57 2020

@author: yakup
"""

import numpy as np
import pandas as pd 
import re
import nltk


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

nltk.download("stopwords")
from nltk.corpus import stopwords


#yorumlar=pd.read_table("Restaurant_Reviews.tsv")
yorumlar=pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

satir=yorumlar.index

derlem=[]

for i in range(len(satir)):
    yorum=re.sub("[^a-zA-Z]"," ",yorumlar["Review"][i])
    yorum=yorum.lower().split()
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum=" ".join(yorum)
    derlem.append(yorum)
    
    
#bag of words


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(derlem).toarray()
y=yorumlar.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
gaus=GaussianNB()
gaus.fit(x_train,y_train)

y_tahmin=gaus.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_tahmin)   


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(5,5))
colors=["green",'pink']
pos=yorumlar[yorumlar['Liked']==1]
neg=yorumlar[yorumlar['Liked']==0]
ck=[pos['Liked'].count(),neg['Liked'].count()]
legpie=plt.pie(ck,labels=["Positive","Negative"],
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors = colors,
                 startangle = 45,
                 explode=(0, 0.1))


stop=stopwords.words('english')

from wordcloud import WordCloud
positivedata = yorumlar[ yorumlar['Liked'] == 1]
positivedata =positivedata['Review']
negdata = yorumlar[yorumlar['Liked'] == 0]
negdata= negdata['Review']

def wordcloud_draw(yorumlar, color = 'white'):
    words = ' '.join(yorumlar)
    cleaned_word = " ".join([word for word in words.split()
                              if(word!='food' and word!='place')
                            ])
    wordcloud = WordCloud(stopwords=stop,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    

wordcloud_draw(positivedata,'white')

wordcloud_draw(negdata,"white")


















    
             
               
