from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

zipcode = np.array([10001, 10002, 10003, 10004, 10005]).reshape(-1, 1)
date_sold = np.array([20250101, 20250201, 20250301, 20250401, 20250501]).reshape(-1, 1)
square_feet = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)

prices = np.array([150000, 200000, 250000, 300000, 350000])

X = np.hstack((zipcode, date_sold, square_feet))
y = prices
# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)  # This is where learning happens

# Step 5: Make predictions
new_house = np.array([[11212, 20250601, 1200]])
predicted_price = model.predict(new_house)
print(f"predicted price for new house with zipcode 11212, date sold 2025-06-01, and 1200 square feet: ${predicted_price[0]:,.2f}")

# Step 6: Evaluate the model
score = model.score(X_test, y_test)
print(f"Model accuracy (R-squared): {score:.2f}")