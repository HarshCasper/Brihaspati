import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'headbrain.csv')
X=dataset["Head Size(cm^3)"].values
Y=dataset["Brain Weight(grams)"].values

print("The mean of the Head Size Values is %r" %(np.mean(X)))
print("The mean of the Brain Weight Values is %r" %(np.mean(Y)))
print("The total number of values available are %r" %(len(X)))

mean_X=np.mean(X)
mean_Y=np.mean(Y)
length=len(X)
numerator=0
denominator=0
for i in range(length):
    numerator+=(X[i]-mean_X)*(Y[i]-mean_Y)
    denominator+=(X[i]-mean_X)**2
b1=numerator/denominator
b0=mean_Y-(b1*mean_X)
print("The two coefficients are %r and %r" %(b1,b0))

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()
