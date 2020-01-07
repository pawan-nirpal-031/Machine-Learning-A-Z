import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ds = pd.read_csv('50_Startups.csv')
x = ds.iloc[:,:-1].values #independent  vars or features used to predict profit
y = ds.iloc[:,4] #dependent var

# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
lab_en = LabelEncoder()
x[:,3] = lab_en.fit_transform(x[:,3])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.int64)
y = np.array(y,dtype = np.int64)
#avoiding dummy var trap (Redundant dependency)
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)


#model fitting or training
from sklearn.linear_model import LinearRegression as LR
reg = LR()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
y_pred = np.array(y_pred,dtype = np.int64)