"""
Exercise 01: Linear Regression from Scratch - SOLUTION
=======================================================

SETUP INSTRUCTIONS:
-------------------
1. Create a virtual environment:
   python3 -m venv venv

2. Activate the virtual environment:
   - On macOS/Linux: source venv/bin/activate
   - On Windows: venv\Scripts\activate

3. Install required packages:
   pip install -r requirements.txt

4. Run the program:
   python3 ex1-main.py

5. Graphs will open on a new window and saved as training_loss.png, final_fit.png, weight_evolution.png, and bias_evolution.png

Complete implementation of gradient descent for linear regression.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: DATA AND BUILDING BLOCKS
# =============================================================================

np.random.seed(42)
X = np.random.uniform(0, 10, 100)
y = 2.5 * X + 7 + np.random.normal(0, 2, 100)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.title('Training Data')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.show()

def predict(X, w, b):
    return w * X + b

def compute_mse(y_true, y_pred):
    n = len(y_true)
    return (1/n) * np.sum((y_pred - y_true) ** 2)

def compute_gradients(X, y_true, y_pred):
    n = len(X)
    error = y_pred - y_true
    dw = (2/n) * np.sum(error * X)
    db = (2/n) * np.sum(error)
    return dw, db

# =============================================================================
# PART 2: TRAINING LOOP
# =============================================================================
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """Train linear regression using gradient descent."""
    # Initialize weights
    w = np.random.randn() * 0.01
    b = 0.0
    
    history = {'loss': [], 'w': [], 'b': []}
    
    for epoch in range(epochs):
        # Step 1: Make predictions
        y_pred = predict(X, w, b)
        
        # Step 2: Compute loss
        loss = compute_mse(y, y_pred)
        
        # Step 3: Compute gradients
        dw, db = compute_gradients(X, y, y_pred)
        
        # Step 4: Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record history
        history['loss'].append(loss)
        history['w'].append(w)
        history['b'].append(b)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b, history

# =============================================================================
# PART 3: TRAIN AND VISUALIZE
# =============================================================================


if __name__ == "__main__":
    print("Training Linear Regression from Scratch...")
    print("=" * 50)
    
    w_final, b_final, history = gradient_descent(X, y, learning_rate=0.01, epochs=1000)
    
    print("\n" + "=" * 50)
    print(f"Final Model: y = {w_final:.4f} * x + {b_final:.4f}")
    print(f"True Relationship: y = 2.5 * x + 7")
    
    # Plot the loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], 'b-', linewidth=2)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot the final fit
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, edgecolors='black', linewidth=0.5, label='Data')
    
    # Regression line
    X_line = np.linspace(0, 10, 100)
    y_line = predict(X_line, w_final, b_final)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label=f'Learned: y = {w_final:.2f}x + {b_final:.2f}')
    
    # True line
    y_true_line = 2.5 * X_line + 7
    plt.plot(X_line, y_true_line, 'g--', linewidth=2, label='True: y = 2.5x + 7')
    
    plt.title('Linear Regression from Scratch')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Weight evolution animation
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['w'], label='w')
    plt.axhline(y=2.5, color='g', linestyle='--', label='True w=2.5')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['b'], label='b')
    plt.axhline(y=7, color='g', linestyle='--', label='True b=7')
    plt.xlabel('Epoch')
    plt.ylabel('Bias')
    plt.title('Bias Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # =============================================================================
    # PART 4: COMPARE WITH SKLEARN
    # =============================================================================
    
    from sklearn.linear_model import LinearRegression
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X.reshape(-1, 1), y)
    
    print("\n" + "=" * 50)
    print("COMPARISON WITH SKLEARN")
    print("=" * 50)
    print(f"Your Model:    w = {w_final:.4f}, b = {b_final:.4f}")
    print(f"sklearn Model: w = {sklearn_model.coef_[0]:.4f}, b = {sklearn_model.intercept_:.4f}")
    print(f"True Values:   w = 2.5000, b = 7.0000")
    
    # =============================================================================
    # REFLECTION ANSWERS
    # =============================================================================
    
    print("\n" + "=" * 50)
    print("REFLECTION ANSWERS")
    print("=" * 50)
    
    print("\nQ1: What happens with learning_rate = 0.1? learning_rate = 1.0?")
    print("Answer(0.1): The training is faster but the model will likely overcorrect in an attempt to find the optimal value.")
    print("Answer(1.0): The minimal loss returned from the training would increase because overshooting is even more extreme in this case.")
    
    print("\nQ2: What happens with learning_rate = 0.0001?")
    print("Answer: The training would be much slower. It would also be less accurate due to the small step size. 100 epochs would not provide enough iterations to converge to the optimal value.")
    
    print("\nQ3: Why small random initialization instead of zeros?")
    print("Answer: Because Linear Regression does not require randomization due to the linear slope zeros would work fine. However, its good practice to use small values for neural networks to avoid symmetry problems.")
    
    print("\nQ4: How close did you get to true values (w=2.5, b=7)?")
    print("Answer: Very close:")
    print(f"w: {w_final:.4f} vs 2.5 (off by ~{abs(w_final - 2.5):.4f})")
    print(f"b: {b_final:.4f} vs 7 (off by ~{abs(b_final - 7):.4f})")