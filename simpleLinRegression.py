import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd

#reading the dataset with pandas
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression as LR
reg = LR()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

plt.scatter(x_train,y_train,c ='r')
plt.plot(x_train,reg.predict(x_train),c = 'b')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.title("Salary vs experiance in training set")
plt.show()

plt.scatter(x_test,y_test,c ='r')
plt.plot(x_train,reg.predict(x_train),c = 'b')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.title("Salary vs experiance in test set")
plt.show()