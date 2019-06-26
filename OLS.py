import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X= np.random.randn(100)
y= 9*X + 18 +np.random.randn(100)

plt.scatter(X,y)
plt.show()

X=np.c_[X,np.ones(100)] #x convert into matrix with nxt column having 100 rows

theta= np.linalg.inv(X.T @ X)@(X.T @ y)




