import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('bank-full.csv',sep=";", na_values = 'unknown')

X=dataset.iloc[:,0:16].values
y=dataset.iloc[:,-1].values



from sklearn.preprocessing import Imputer
imp = Imputer(strategy="mean") # read info for imputer. this is must.

X[:,[0,5,9,11,12,13,14]] = imp.fit_transform(X[:,[0,5,9,11,12,13,14]])


test = pd.DataFrame(X[:,[1,2,3,4,6,7,8,10,15]])
test[0].value_counts()
test[1].value_counts()
test[2].value_counts()
test[3].value_counts()
test[4].value_counts()
test[5].value_counts()
test[6].value_counts()
test[7].value_counts()
test[8].value_counts()

test[0] = test[0].fillna('blue-collar')
test[1] = test[1].fillna('married')
test[2] = test[2].fillna('secondary')
test[3] = test[3].fillna('no')
test[4] = test[4].fillna('yes')
test[5] = test[5].fillna('no')
test[6] = test[6].fillna('cellular')
test[7] = test[7].fillna('may')
test[8] = test[8].fillna('failure')

X[:,[1,2,3,4,6,7,8,10,15]] = test

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

X[:,1] = lab.fit_transform(X[:,1])
X[:,2] = lab.fit_transform(X[:,2])
X[:,3] = lab.fit_transform(X[:,3])
X[:,4] = lab.fit_transform(X[:,4])
X[:,6] = lab.fit_transform(X[:,6])
X[:,7] = lab.fit_transform(X[:,7])
X[:,8] = lab.fit_transform(X[:,8])
X[:,10] = lab.fit_transform(X[:,10])
X[:,15] = lab.fit_transform(X[:,15])

y = lab.fit_transform(y)
#lab.classes_

#plt.scatter(X[0],y)
#plt.show()


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features=[1,2,3,4,6,7,8,10,15])
X = one.fit_transform(X)
#X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


pd.plotting.scatter_matrix(dataset)






