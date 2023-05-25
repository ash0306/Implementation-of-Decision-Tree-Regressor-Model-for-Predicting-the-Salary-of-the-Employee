# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and get dataset info
2. check for null values
3. Map values for position column
4. Split train data and test data
5. Import decision tree regressor and fit it for data
6. Calculate mse r2 and y_predict

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 212220040020
RegisterNumber:  Aswathi S
*/

import pandas as pd
df=pd.read_csv('/content/Salary.csv')
print('1) df.head()')
df.head()
print('2) df.info')
df.info
print('3) df.isnull().sum()')
df.isnull().sum()
print('4) df.head() for salary')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()
print('5) MSE value')
x=df[["Position","Level"]]
y=df[["Salary"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
print('6) r2 value')
r2=metrics.r2_score(y_test,y_pred)
r2
print('7) Data prediction')
dt.predict([[5,6]])

```

## Output:
![1](images/1.png)\
![2](images/2.png)\
![3](images/3.png)\
![4](images/4.png)\
![5](images/5.png)\
![6](images/6.png)\
![7](images/7.png)\


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.