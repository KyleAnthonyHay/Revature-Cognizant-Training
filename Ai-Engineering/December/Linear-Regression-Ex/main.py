import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Linear-Regressions.csv')

X = df[['Hours']]
y = df['Score']

model = LinearRegression()
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_

plt.scatter(X, y, color='blue', label='Students')

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', label=f"y = {w:.3f}x + {b:.2f}")

plt.xlabel('Hours')
plt.ylabel('Score')
plt.legend()
plt.savefig('student-plot.png')
print("Plot saved as student-plot.png")
