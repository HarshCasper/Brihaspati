import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'social.csv')
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,Y_train)
Y_pred=LR.predict(X_test)

from sklearn.metrics import classification_report
print("The classification report is as follows: \n" )
print(classification_report(Y_test,Y_pred))
