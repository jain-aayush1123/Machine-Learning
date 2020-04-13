import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
import tensorflow

import os 
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'student-mat.csv')

#read all data
data = pd.read_csv(my_file, sep=";")
#get relevant columns
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]
# print(data.head())

#we are going to predict g3 from data
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#we will split the total data into 9:1 ration. 9 parts to train and 1 to test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)
pickle_in = pickle.load 
print("Co \n", linear.coef_)
print("intercept \n", linear.intercept_)
