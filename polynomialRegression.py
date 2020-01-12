import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ds = pd.read_csv("Position_Salaries.csv")

x = ds.iloc[:,1:2].values
y = ds.iloc[:,2].values

from sklearn.linear_model import LinearRegression as LR# comparison purposes
linreg1 = LR()
linreg2 = LR()
linreg1.fit(x,y)
y_pred1 = linreg1.predict(x)
y_pred1 = np.array(y_pred1,dtype = 'int64')

from sklearn.preprocessing import PolynomialFeatures as PF# polynomial regression obj
polyreg = PF(degree = 2)
x_poly = polyreg.fit_transform(x)# adding new features like x0
x_poly = np.array(x_poly,dtype = 'int64')
polyreg.fit(x_poly,y)
linreg2.fit(x_poly,y)
y_pred2 = linreg2.predict(polyreg.fit_transform(x))

#plotting the results
plt.scatter(x,y,c = 'r')
plt.plot(x,y_pred1,c = 'b')# plotting the linear model which is bad
plt.plot(x,y_pred2,c = 'g')
plt.xlabel("Position")
plt.ylabel("Salary")


plt.show()
