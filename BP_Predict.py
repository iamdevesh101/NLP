import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_excel('blood.xlsx')
#X = dataset.iloc[:,1].values
#y= dataset.iloc[:,-1].values

X = dataset.iloc[2:,1].values
y= dataset.iloc[2:,-1].values # outlier removal by removing column here we are observing result after removing outlier


X = X.reshape(-1,1)

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)     # error because X should be matrix

lin_reg.score(X,y) # gives accuracy


plt.scatter(X,y)

#plt.plot(X, lin_reg.predict(X),c ="r")
plt.scatter(X, lin_reg.predict(X),c ="r")
plt.show()


lin_reg.coef_
lin_reg.intercept_



lin_reg.predict([[21]])
 #predict BP of person age 21










