"""
Demo 05: Activation Function Explorer
=====================================
Week 1, Wednesday - Neural Network Fundamentals

Interactive visualization comparing activation functions.
Shows why non-linearity is crucial for deep learning.

INSTRUCTOR NOTES:
- Reference the Written Content on activation functions
- Focus on ReLU - it's what trainees will use most
- Show the gradient plots to preview backpropagation concepts

Estimated Time: 15-20 minutes
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: WHY ACTIVATION FUNCTIONS?
# =============================================================================
print("=" * 60)
print("DEMO 05: ACTIVATION FUNCTION EXPLORER")
print("The Key to Non-Linear Learning")
print("=" * 60)

print("""
WITHOUT ACTIVATION FUNCTIONS:
    Layer 1: z1 = W1 * x + b1
    Layer 2: z2 = W2 * z1 + b2
    
    Combined: z2 = W2 * (W1 * x + b1) + b2
             z2 = (W2 * W1) * x + (W2 * b1 + b2)
             z2 = W_combined * x + b_combined
    
    Three layers collapse into ONE linear transformation!
    
WITH ACTIVATION FUNCTIONS:
    Each layer adds non-linearity.
    The network can learn complex, non-linear patterns.
""")

# =============================================================================
# SECTION 2: DEFINE ACTIVATION FUNCTIONS
# =============================================================================

def step(z):
    """Step function (original perceptron)."""
    z = np.asarray(z)
    return (z >= 0).astype(float)

def sigmoid(z):
    """Sigmoid / Logistic function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def tanh_activation(z):
    """Hyperbolic tangent."""
    return np.tanh(z)

def relu(z):
    """Rectified Linear Unit."""
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU."""
    return np.where(z > 0, z, alpha * z)

def elu(z, alpha=1.0):
    """Exponential Linear Unit."""
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

# Derivatives (for gradient visualization)
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_grad(z):
    return 1 - np.tanh(z)**2

def relu_grad(z):
    return (z > 0).astype(float)

def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

# =============================================================================
# SECTION 3: COMPARE ALL ACTIVATIONS
# =============================================================================
print("\n--- ACTIVATION FUNCTIONS COMPARISON ---")

z = np.linspace(-5, 5, 1000)

activations = {
    'Step': step(z),
    'Sigmoid': sigmoid(z),
    'Tanh': tanh_activation(z),
    'ReLU': relu(z),
    'Leaky ReLU': leaky_relu(z),
}

# Plot all activations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

colors = ['#E91E63', '#9C27B0', '#3F51B5', '#4CAF50', '#FF9800']

for idx, (name, values) in enumerate(activations.items()):
    ax = axes[idx]
    ax.plot(z, values, color=colors[idx], linewidth=2.5, label=name)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-5, 5)
    ax.set_xlabel('z (input)', fontsize=11)
    ax.set_ylabel('f(z) (output)', fontsize=11)
    ax.set_title(f'{name} Activation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

# Summary in last subplot
ax = axes[5]
for idx, (name, values) in enumerate(activations.items()):
    ax.plot(z, values, color=colors[idx], linewidth=2, label=name, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlim(-5, 5)
ax.set_xlabel('z (input)', fontsize=11)
ax.set_ylabel('f(z) (output)', fontsize=11)
ax.set_title('All Activations Compared', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_activations_overview.png', dpi=100)
plt.show()
print("[Saved: 05_activations_overview.png]")

# =============================================================================
# SECTION 4: ACTIVATION VALUES TABLE
# =============================================================================
print("\n--- ACTIVATION VALUES AT KEY POINTS ---")

test_points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

print(f"{'z':<8}", end="")
for name in ['Step', 'Sigmoid', 'Tanh', 'ReLU', 'L-ReLU']:
    print(f"{name:<10}", end="")
print()
print("-" * 60)

for z_val in test_points:
    print(f"{z_val:<8.1f}", end="")
    print(f"{step(z_val):<10.3f}", end="")
    print(f"{sigmoid(z_val):<10.3f}", end="")
    print(f"{tanh_activation(z_val):<10.3f}", end="")
    print(f"{relu(z_val):<10.3f}", end="")
    print(f"{leaky_relu(z_val):<10.3f}")

# =============================================================================
# SECTION 5: GRADIENTS (PREVIEW BACKPROPAGATION)
# =============================================================================
print("\n--- ACTIVATION GRADIENTS ---")
print("Gradients tell us how fast the output changes with input.")
print("Small gradients = slow learning (vanishing gradient problem)")

# Plot gradients
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sigmoid gradient
ax = axes[0, 0]
ax.plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid')
ax.plot(z, sigmoid_grad(z), 'r--', linewidth=2, label='Gradient')
ax.fill_between(z, 0, sigmoid_grad(z), alpha=0.2, color='red')
ax.set_title('Sigmoid: Vanishing Gradient Problem', fontsize=12)
ax.set_xlabel('z')
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate('Gradient nearly 0\nat extremes!', xy=(3, 0.05), fontsize=10,
            ha='center', color='red')

# Tanh gradient
ax = axes[0, 1]
ax.plot(z, tanh_activation(z), 'b-', linewidth=2, label='Tanh')
ax.plot(z, tanh_grad(z), 'r--', linewidth=2, label='Gradient')
ax.fill_between(z, 0, tanh_grad(z), alpha=0.2, color='red')
ax.set_title('Tanh: Still Has Vanishing Gradient', fontsize=12)
ax.set_xlabel('z')
ax.legend()
ax.grid(True, alpha=0.3)

# ReLU gradient
ax = axes[1, 0]
ax.plot(z, relu(z), 'b-', linewidth=2, label='ReLU')
ax.plot(z, relu_grad(z), 'r--', linewidth=2, label='Gradient')
ax.fill_between(z, 0, relu_grad(z), alpha=0.2, color='red')
ax.set_title('ReLU: Constant Gradient (No Vanishing!)', fontsize=12)
ax.set_xlabel('z')
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate('Gradient = 1\nfor z > 0', xy=(2, 0.5), fontsize=10, ha='center')
ax.annotate('Dying ReLU:\nGradient = 0\nfor z < 0', xy=(-3, 0.5), fontsize=10,
            ha='center', color='red')

# Leaky ReLU gradient
ax = axes[1, 1]
ax.plot(z, leaky_relu(z), 'b-', linewidth=2, label='Leaky ReLU')
ax.plot(z, leaky_relu_grad(z), 'r--', linewidth=2, label='Gradient')
ax.fill_between(z, 0, leaky_relu_grad(z), alpha=0.2, color='red')
ax.set_title('Leaky ReLU: No Dead Neurons!', fontsize=12)
ax.set_xlabel('z')
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate('Small gradient\nfor z < 0', xy=(-3, 0.3), fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('05_gradients.png', dpi=100)
plt.show()
print("[Saved: 05_gradients.png]")

# =============================================================================
# SECTION 6: SIGMOID VS RELU DEEP DIVE
# =============================================================================
print("\n--- SIGMOID vs ReLU ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sigmoid characteristics
ax = axes[0]
z_range = np.linspace(-10, 10, 500)
ax.plot(z_range, sigmoid(z_range), 'b-', linewidth=2.5)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)

# Annotate
ax.annotate('Output bounded [0, 1]', xy=(0.5, 0.5), xytext=(4, 0.5),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('Saturates at 0', xy=(-6, 0.01), fontsize=10, color='red')
ax.annotate('Saturates at 1', xy=(4, 0.99), fontsize=10, color='red')

ax.set_xlim(-10, 10)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('z', fontsize=12)
ax.set_ylabel('sigmoid(z)', fontsize=12)
ax.set_title('Sigmoid: Smooth but Saturates', fontsize=13)
ax.grid(True, alpha=0.3)

# ReLU characteristics
ax = axes[1]
ax.plot(z_range, relu(z_range), 'g-', linewidth=2.5)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)

# Annotate
ax.annotate('Linear for z > 0', xy=(5, 5), xytext=(2, 7),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('Dead neurons\nwhen z < 0', xy=(-5, 0), fontsize=10, color='red',
            ha='center')

ax.set_xlim(-10, 10)
ax.set_ylim(-1, 10)
ax.set_xlabel('z', fontsize=12)
ax.set_ylabel('ReLU(z)', fontsize=12)
ax.set_title('ReLU: Simple but Powerful', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_sigmoid_vs_relu.png', dpi=100)
plt.show()
print("[Saved: 05_sigmoid_vs_relu.png]")

# =============================================================================
# SECTION 7: CHOOSING ACTIVATION FUNCTIONS
# =============================================================================
print("\n--- CHOOSING ACTIVATION FUNCTIONS ---")

print("""
LAYER POSITION -> ACTIVATION CHOICE:

+----------------------+-------------------+---------------------------+
| Layer Type           | Best Activation   | Why?                      |
+----------------------+-------------------+---------------------------+
| Hidden Layers        | ReLU              | Fast, no vanishing grad   |
| (Default)            | Leaky ReLU        | If dying ReLU is issue    |
+----------------------+-------------------+---------------------------+
| Output: Binary Class | Sigmoid           | Outputs probability [0,1] |
+----------------------+-------------------+---------------------------+
| Output: Multi-Class  | Softmax           | Outputs prob distribution |
+----------------------+-------------------+---------------------------+
| Output: Regression   | Linear (None)     | Can output any value      |
+----------------------+-------------------+---------------------------+
| RNNs                 | Tanh              | Bounded, zero-centered    |
+----------------------+-------------------+---------------------------+
""")

# =============================================================================
# SECTION 8: INTERACTIVE EXAMPLE
# =============================================================================
print("\n--- EFFECT ON NEURAL NETWORK OUTPUT ---")

# Simulate a small network with different activations
np.random.seed(42)

# Input
x = np.array([0.5, -0.3, 0.8])
W = np.random.randn(3, 4) * 0.5
b = np.zeros(4)

# Linear combination
z = np.dot(x, W) + b

print(f"Input: {x}")
print(f"After weights (z = Wx + b): {z}")
print()
print("After different activations:")
print(f"  Step:       {step(z)}")
print(f"  Sigmoid:    {sigmoid(z)}")
print(f"  Tanh:       {tanh_activation(z)}")
print(f"  ReLU:       {relu(z)}")
print(f"  Leaky ReLU: {leaky_relu(z)}")

# =============================================================================
# SECTION 9: KEY TAKEAWAYS
# =============================================================================
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. PURPOSE OF ACTIVATION FUNCTIONS:
   - Introduce non-linearity
   - Without them, deep networks collapse to linear models

2. SIGMOID: 
   - Output [0, 1] (probability interpretation)
   - Problem: Vanishing gradients at extremes
   - Use: Binary classification OUTPUT layer

3. TANH:
   - Output [-1, 1] (zero-centered)
   - Problem: Still has vanishing gradients
   - Use: RNNs, some hidden layers

4. RELU (Rectified Linear Unit):
   - Output [0, infinity)
   - No vanishing gradient for positive values
   - Problem: "Dying ReLU" (neurons stuck at 0)
   - Use: DEFAULT for hidden layers

5. LEAKY RELU:
   - Fixes dying ReLU with small negative slope
   - Use: When ReLU causes dead neurons

6. RULE OF THUMB:
   Hidden layers: ReLU
   Output (binary): Sigmoid
   Output (multi-class): Softmax
   Output (regression): Linear
""")

print("Next: Forward propagation - tracing values through a network!")

