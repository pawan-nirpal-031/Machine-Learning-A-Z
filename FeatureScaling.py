import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer #missing data handler library
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # the raw data may have some missing features so this problem could 
#be fixed by either removing them alltogather but by doing that we might lose valuable info so missing one's are replaced by mean of that colomn

#reading the dataset with pandas
dataset = pd.read_csv('data.csv')
x = dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

#fixing the missing data by mean of colomns
imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:, 1:3])

labelen_x = LabelEncoder() # string to numeric encoding object



# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#splitting train and test data
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler as stdsc
sc_x = stdsc()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)




