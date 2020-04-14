import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
import os
from sklearn.utils import shuffle
import statistics

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'home_data.csv')

# * returns dataframe of data
# * make sure to check for separator
data = pd.read_csv(my_file, sep=",")
data = data[['bedrooms', 'bathrooms', 'sqft_living',
             'sqft_lot', 'floors', 'zipcode', 'price']]

predict = "price"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# * return top 5 rows
# print(data.head())

# * serching through data
# seattle_houses = []
# for i in data.index:
#     if(data["zipcode"][i] == 98039):
#         seattle_houses.append(data['price'][i])
# print("size: ", statistics.mean(seattle_houses))


# * filtering with pandas
# bool_series = data["sqft_living"].between(2000, 4000, inclusive=True)
# print("len: ", len(data[bool_series])/len(data["sqft_living"]))


# * scatter plot
# * x axis is characterstics and y is the dependent variable
# pyplot.scatter(data.sqft_living, data.price)
# pyplot.show()

# * splitting data
# * id we give random_state attr to split, it fixes the seed for randomization to give same random answer everytime
x_training_dataset, x_test_dataset, y_training_dataset, y_test_dataset = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2)

# * making model
features_model = linear_model.LinearRegression().fit(
    x_training_dataset, y_training_dataset)
acc = features_model.score(x_test_dataset, y_test_dataset)
print("acc: ", acc)
# print(data.size)
# print(data[predict].mean())

# * plotting actual values against real values
# pyplot.plot(x_test_dataset, y_test_dataset, '.',
#             x_test_dataset, features_model.predict(x_test_dataset), '-'
#             )
# pyplot.show()

# * getting parameters
print("coeffs: ", features_model.coef_)
print("intercept: ", features_model.intercept_)

# * rmse
rmse = sklearn.metrics.mean_squared_error(
    y_test_dataset, features_model.predict(x_test_dataset), squared=False)

print("rmse: ", rmse)
