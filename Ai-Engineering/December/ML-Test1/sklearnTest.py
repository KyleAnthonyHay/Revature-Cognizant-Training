from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=80
)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model accuracy (R-squared): {score:.2f}")

# df = pd.DataFrame(housing.data, columns=housing.feature_names)
# print(df)
#         MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
# 0      8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
print(f"Target List: {housing.target}")
newHouse = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
predictedPrice = model.predict(newHouse)
print(f"Predicted price for new house: ${predictedPrice[0]:,.2f} Million")