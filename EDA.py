import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
data = np.genfromtxt('eda-18-ass1-data.txt')
data = pd.DataFrame(data)
data_x = data.iloc[:,1:6]
data_y = data.iloc[:,6:7]

#regression function define

def regression(x0, X, Y, C):
    
     
   
    xw = X.T * weight(x0, X, C)
    
    #expression for Beta value to minize the function error 
    beta = np.linalg.pinv(xw @ np.array(X)) @ xw @ np.array(Y)
    
    #pretiction output
    return x0 @ beta

#weight fuction

def weight(x0, X, C):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * C * C))
X_train,X_test,Y_train,Y_test = train_test_split(data_x,data_y)

#defined empty array to store predicted value
lst =[]
for i in  range(len(X_test)):
	lst.append(regression(X_test.iloc[i,:],X_train,Y_train, C=1)) #taken c=1
predict = pd.Series(lst)

Y_test = Y_test.reset_index(drop=True)
predict.to_csv('Predict.csv')
X_test = X_test.reset_index(drop=True)
X_test.to_csv('X_test.csv')
Y_test.to_csv('Y_true.csv')
