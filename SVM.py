import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Support-Vector-Regression-Data.csv')


x = df.x.values.reshape(-1, 1)
y = df.y.values.reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.40, random_state=42)


SupportVectoRegModel = SVR()
SupportVectoRegModel.fit(x_train, y_train)


y_pred = SupportVectoRegModel.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)
