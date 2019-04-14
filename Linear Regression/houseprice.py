# Import Dependencies

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston dataset

boston=datasets.load_boston()

# X - feature vectors
# y - Target values

X=boston.data
y=boston.target

# splitting X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)

# Create linear regression objest

lin_reg=linear_model.LinearRegression()

# Train the model using trai and test data

lin_reg.fit(X_train,y_train)

# Presict values for X_test data

predicted = lin_reg.predict(X_test)

# Regression coefficients

print('Coefficients are:\n',lin_reg.coef_)

# Intecept

print('\nIntercept : ',lin_reg.intercept_)

# variance score: 1 means perfect prediction

print('Variance score: ',lin_reg.score(X_test, y_test))

# Mean Squared Erroe

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, predicted))

# Original data of X_test

expected = y_test

# Plot a graph for expected and predicted values

plt.title('ActualPrice Vs PredictedPrice (BOSTON Housing Dataset)')
plt.scatter(expected,predicted,c='b',marker='.',s=36)
plt.plot([0, 50], [0, 50], '--r')
plt.xlabel('Actual Price(1000$)')
plt.ylabel('Predicted Price(1000$)')
plt.show()
