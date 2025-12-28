"""
Demo 04: Perceptron Learning Logic Gates
========================================
Week 1, Wednesday - Neural Network Fundamentals

This demo shows how a single perceptron learns AND/OR logic gates.
Trainees will see weight updates and decision boundary evolution.

INSTRUCTOR NOTES:
- Start with the perceptron diagram to show the components
- This is the ATOMIC unit of neural networks
- Show why XOR fails (the "aha!" moment for multi-layer networks)

Estimated Time: 20-25 minutes
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: THE PERCEPTRON
# =============================================================================
print("=" * 60)
print("DEMO 04: PERCEPTRON LEARNING LOGIC GATES")
print("The Building Block of Neural Networks")
print("=" * 60)

print("""
A PERCEPTRON computes:
    
    z = w1*x1 + w2*x2 + b    (weighted sum + bias)
    y = 1 if z >= 0 else 0   (step activation)

We'll teach it to learn AND and OR gates!
""")

# =============================================================================
# SECTION 2: LOGIC GATE DATA
# =============================================================================
print("\n--- LOGIC GATE TRUTH TABLES ---")

# AND gate: output 1 only if BOTH inputs are 1
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# OR gate: output 1 if ANY input is 1
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# XOR gate: output 1 if inputs are DIFFERENT
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("AND Gate:        OR Gate:         XOR Gate:")
print("x1 x2 | y        x1 x2 | y        x1 x2 | y")
print("-" * 50)
for i in range(4):
    print(f"{X_and[i,0]}  {X_and[i,1]}  | {y_and[i]}        "
          f"{X_or[i,0]}  {X_or[i,1]}  | {y_or[i]}        "
          f"{X_xor[i,0]}  {X_xor[i,1]}  | {y_xor[i]}")

# =============================================================================
# SECTION 3: PERCEPTRON CLASS
# =============================================================================

class Perceptron:
    """A single perceptron (binary classifier)."""
    
    def __init__(self, n_features, learning_rate=0.1):
        """Initialize with small random weights."""
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.history = []  # Track learning progress
    
    def step_activation(self, z):
        """Step function: 1 if z >= 0, else 0."""
        return (z >= 0).astype(int)
    
    def predict(self, X):
        """Forward pass: compute predictions."""
        z = np.dot(X, self.weights) + self.bias
        return self.step_activation(z)
    
    def train(self, X, y, epochs=20):
        """Train using perceptron learning rule."""
        print(f"\nTraining for {epochs} epochs...")
        print(f"Initial weights: {self.weights}, bias: {self.bias:.3f}")
        
        for epoch in range(epochs):
            errors = 0
            
            for xi, yi in zip(X, y):
                # Prediction
                prediction = self.predict(xi.reshape(1, -1))[0]
                
                # Error
                error = yi - prediction
                
                if error != 0:
                    errors += 1
                    # Update rule: w = w + learning_rate * error * x
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
            
            # Record history
            self.history.append({
                'epoch': epoch,
                'weights': self.weights.copy(),
                'bias': self.bias,
                'errors': errors
            })
            
            if epoch < 5 or errors == 0:
                print(f"  Epoch {epoch}: errors={errors}, "
                      f"w={self.weights}, b={self.bias:.3f}")
            
            if errors == 0:
                print(f"\nConverged at epoch {epoch}!")
                break
        
        return self

# =============================================================================
# SECTION 4: TRAIN ON AND GATE
# =============================================================================
print("\n--- TRAINING: AND GATE ---")

perceptron_and = Perceptron(n_features=2, learning_rate=0.1)
perceptron_and.train(X_and, y_and, epochs=20)

# Test predictions
print("\nAND Gate Results:")
print(f"Learned weights: w1={perceptron_and.weights[0]:.3f}, w2={perceptron_and.weights[1]:.3f}")
print(f"Learned bias: {perceptron_and.bias:.3f}")
print(f"Decision boundary: {perceptron_and.weights[0]:.3f}*x1 + {perceptron_and.weights[1]:.3f}*x2 + {perceptron_and.bias:.3f} = 0")

predictions = perceptron_and.predict(X_and)
print(f"\nPredictions: {predictions}")
print(f"Actual:      {y_and}")
print(f"Correct: {(predictions == y_and).all()}")

# =============================================================================
# SECTION 5: VISUALIZE DECISION BOUNDARY
# =============================================================================
print("\n--- VISUALIZING DECISION BOUNDARIES ---")

def plot_decision_boundary(perceptron, X, y, title, filename):
    """Plot data points and decision boundary."""
    plt.figure(figsize=(8, 6))
    
    # Plot points
    for label, marker, color in [(0, 'o', 'red'), (1, 's', 'green')]:
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, 
                   s=150, label=f'Class {label}', edgecolors='black', linewidth=2)
    
    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # Solving for x2: x2 = -(w1*x1 + b) / w2
    w1, w2 = perceptron.weights
    b = perceptron.bias
    
    if abs(w2) > 1e-6:  # Avoid division by zero
        x1_range = np.linspace(-0.5, 1.5, 100)
        x2_boundary = -(w1 * x1_range + b) / w2
        
        plt.plot(x1_range, x2_boundary, 'b-', linewidth=2, label='Decision Boundary')
        
        # Shade regions
        plt.fill_between(x1_range, x2_boundary, 2, alpha=0.1, color='green')
        plt.fill_between(x1_range, x2_boundary, -1, alpha=0.1, color='red')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.show()
    print(f"[Saved: {filename}]")

plot_decision_boundary(perceptron_and, X_and, y_and, 
                       'AND Gate: Perceptron Decision Boundary',
                       '04_and_gate.png')

# =============================================================================
# SECTION 6: TRAIN ON OR GATE
# =============================================================================
print("\n--- TRAINING: OR GATE ---")

perceptron_or = Perceptron(n_features=2, learning_rate=0.1)
perceptron_or.train(X_or, y_or, epochs=20)

predictions = perceptron_or.predict(X_or)
print(f"\nPredictions: {predictions}")
print(f"Actual:      {y_or}")
print(f"Correct: {(predictions == y_or).all()}")

plot_decision_boundary(perceptron_or, X_or, y_or,
                       'OR Gate: Perceptron Decision Boundary',
                       '04_or_gate.png')

# =============================================================================
# SECTION 7: THE XOR PROBLEM
# =============================================================================
print("\n--- THE XOR PROBLEM ---")
print("Now let's try XOR - this is where single perceptrons FAIL!")

perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
perceptron_xor.train(X_xor, y_xor, epochs=100)  # Even 100 epochs won't help

predictions = perceptron_xor.predict(X_xor)
print(f"\nPredictions: {predictions}")
print(f"Actual:      {y_xor}")
print(f"Correct: {(predictions == y_xor).all()}")

# Visualize the problem
plt.figure(figsize=(8, 6))
colors = ['red' if yi == 0 else 'green' for yi in y_xor]
markers = ['o' if yi == 0 else 's' for yi in y_xor]

for i in range(4):
    plt.scatter(X_xor[i, 0], X_xor[i, 1], c=colors[i], marker=markers[i],
               s=150, edgecolors='black', linewidth=2)

# Show that no single line can separate
plt.text(0.5, -0.3, 'NO single line can separate XOR!', 
         ha='center', fontsize=12, color='red', fontweight='bold')

# Draw some attempted lines
for angle in [45, -45, 90, 0]:
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = np.tan(np.radians(angle)) * (x1 - 0.5) + 0.5
    plt.plot(x1, x2, '--', alpha=0.3, linewidth=1)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.title('XOR Gate: NOT Linearly Separable!', fontsize=14)
plt.grid(True, alpha=0.3)

# Legend
plt.scatter([], [], c='red', marker='o', s=100, label='Class 0', edgecolors='black')
plt.scatter([], [], c='green', marker='s', s=100, label='Class 1', edgecolors='black')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('04_xor_problem.png', dpi=100)
plt.show()
print("[Saved: 04_xor_problem.png]")

# =============================================================================
# SECTION 8: LEARNING VISUALIZATION
# =============================================================================
print("\n--- LEARNING PROGRESS ---")

# Visualize how AND gate weights evolved
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Weight history for AND gate
epochs = [h['epoch'] for h in perceptron_and.history]
w1_history = [h['weights'][0] for h in perceptron_and.history]
w2_history = [h['weights'][1] for h in perceptron_and.history]
b_history = [h['bias'] for h in perceptron_and.history]
error_history = [h['errors'] for h in perceptron_and.history]

axes[0].plot(epochs, w1_history, 'b-', label='w1', linewidth=2)
axes[0].plot(epochs, w2_history, 'r-', label='w2', linewidth=2)
axes[0].plot(epochs, b_history, 'g-', label='bias', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Value')
axes[0].set_title('AND Gate: Weight Evolution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, error_history, 'k-', linewidth=2, marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Errors')
axes[1].set_title('AND Gate: Errors per Epoch')
axes[1].grid(True, alpha=0.3)

# Final decision boundary
axes[2].set_title('AND Gate: Final Solution')
for label, marker, color in [(0, 'o', 'red'), (1, 's', 'green')]:
    mask = y_and == label
    axes[2].scatter(X_and[mask, 0], X_and[mask, 1], c=color, marker=marker,
                   s=150, edgecolors='black', linewidth=2)

w1, w2 = perceptron_and.weights
b = perceptron_and.bias
x1_range = np.linspace(-0.5, 1.5, 100)
x2_boundary = -(w1 * x1_range + b) / w2
axes[2].plot(x1_range, x2_boundary, 'b-', linewidth=2)
axes[2].set_xlim(-0.5, 1.5)
axes[2].set_ylim(-0.5, 1.5)
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_learning_progress.png', dpi=100)
plt.show()
print("[Saved: 04_learning_progress.png]")

# =============================================================================
# SECTION 9: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. PERCEPTRON COMPONENTS:
   - Inputs (features)
   - Weights (learned importance)
   - Bias (threshold shift)
   - Activation (decision function)

2. PERCEPTRON LEARNING RULE:
   w_new = w_old + learning_rate * error * input
   Simple but effective for linearly separable data!

3. LINEAR SEPARABILITY:
   - AND, OR: Linearly separable -> Perceptron succeeds
   - XOR: NOT linearly separable -> Perceptron FAILS

4. THE SOLUTION TO XOR:
   Stack multiple perceptrons into layers!
   This is the Multi-Layer Perceptron (MLP).

5. HISTORICAL NOTE:
   The XOR problem caused the first "AI Winter" in 1969.
   Solution: Multi-layer networks + backpropagation (1986).
""")

print("Next: Activation functions - why non-linearity matters!")

