# model.py

# Importing libraries we'll be using in this class
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import request
import json

# Import Dataset
dataset = pd.read_csv("DATA/Salary_Data.csv")

# Have a look at the dataset's head
print(dataset.head())

# Create variable X and Y for training
# First to last column
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values

print(X)

print(y)

# Split the training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

Model = LinearRegression()

Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)

# Save the model
pickle.dump(Model, open('model.pkl', 'wb'))


# Load the model again
imported_model = pickle.load(open('model.pkl', 'rb'))
print(imported_model.predict([[1.8]]))

""" 
Pickle's open() function:
Character	Meaning
'r'	open for reading (default)
'w'	open for writing, truncating the file first
'x'	create a new file and open it for writing
'a'	open for writing, appending to the end of the file if it exists
'b'	binary mode
't'	text mode (default)
'+'	open a disk file for updating (reading and writing)
'U'	universal newline mode (deprecated) 
"""
