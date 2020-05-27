# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv(' ')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data (If Any)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [ ])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Training The Regression Model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Data Visualisation 

    # Low Resolution
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title(' ')
plt.xlabel(' ')
plt.ylabel(' ')
plt.show()

    # High Resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title(' ')
plt.xlabel(' ')
plt.ylabel(' ')
plt.show()