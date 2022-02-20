#This program is made by Rahul Malhotra(20BCS6026) 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#from sklearn.metrics import r2_score
#importing the datasets
dataset=pd.read_csv("C:/Users/Lenovo/.jupyter/CarPrice_Assignment.csv")
data1=dataset.head(10)
data2=dataset.tail(10)
shape=dataset.shape
#dropping the duplicates
dataset = dataset.drop_duplicates()
#dropping the Null Values
dataset = dataset.dropna()
#dropping the non important independent variables
dataset = dataset.drop(['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'], axis=1)
#analyzing the dataset visually
#drawing the countplot
sns.countplot(x="carlength", data=dataset)
#Plotting the Histogram
dataset["carlength"].plot.hist()
#seperate the dependet and the independent variables
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#dividing the datasets into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=1/3)
#training the linear regression machine learning model using the training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
#now predicting the results for the testing datasets
Y_pred=regressor.predict(X_test)
#testing the accuray of the model using accuracy
#from sklearn.metrics import mean_squared_error
r2_score=regressor.score(X,Y)