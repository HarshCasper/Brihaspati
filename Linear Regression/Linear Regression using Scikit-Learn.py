import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
linear=LinearRegression()

dataset=pd.read_csv(r'headbrain.csv')
X=dataset["Head Size(cm^3)"].values
Y=dataset["Brain Weight(grams)"].values
print("The mean of the Head Size Values is %r" %(np.mean(X)))
print("The mean of the Brain Weight Values is %r" %(np.mean(Y)))
print("The total number of values available are %r" %(len(X)))

length=len(X)
X=X.reshape(length,1)
linear=linear.fit(X,Y)
Y_pred=linear.predict(X)

# Calculating the Root-Mean Squared Error
mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print(rmse)
