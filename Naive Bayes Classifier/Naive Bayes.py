import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

dataset=pd.read_csv(r'train.csv', na_values=['?'])
dataset=dataset.dropna(axis = 0, how ='any') 

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X = dataset[['Column3', 'Column9', 'Column11', 'Column12', 'Column13']]
Y = dataset['Column14']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
model=GaussianNB()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
print(classification_report(Y_test,Y_pred))
