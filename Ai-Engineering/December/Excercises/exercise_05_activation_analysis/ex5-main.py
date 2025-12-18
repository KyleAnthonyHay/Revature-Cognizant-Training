"""
Exercise 05: Activation Function Analysis
=========================================

Implement and analyze activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: IMPLEMENT ACTIVATION FUNCTIONS
# =============================================================================

def step(z):
    """Step function (Heaviside)."""
    # TODO: Return 1 if z >= 0, else 0
    z = np.asarray(z)
    return (z >= 0).astype(float)


def sigmoid(z):
    """Sigmoid / Logistic function."""
    # TODO: Return 1 / (1 + exp(-z))
    # Hint: Use np.clip(z, -500, 500) to avoid overflow
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def tanh_activation(z):
    """Hyperbolic tangent."""
    # TODO: Return tanh(z)
    return np.tanh(z)


def relu(z):
    """Rectified Linear Unit."""
    # TODO: Return max(0, z)
    return np.maximum(0, z)


def leaky_relu(z, alpha=0.01):
    """Leaky ReLU."""
    # TODO: Return z if z > 0, else alpha * z
    return np.where(z > 0, z, alpha * z)


# =============================================================================
# PART 2: IMPLEMENT DERIVATIVES
# =============================================================================

def sigmoid_derivative(z):
    """Derivative of sigmoid."""
    # TODO: sigmoid(z) * (1 - sigmoid(z))
    s = sigmoid(z)
    return s * (1 - s)


def tanh_derivative(z):
    """Derivative of tanh."""
    # TODO: 1 - tanh(z)^2
    return 1 - np.tanh(z)**2


def relu_derivative(z):
    """Derivative of ReLU."""
    # TODO: 1 if z > 0, else 0
    return (z > 0).astype(float)


def leaky_relu_derivative(z, alpha=0.01):
    """Derivative of Leaky ReLU."""
    # TODO: 1 if z > 0, else alpha
    return np.where(z > 0, 1, alpha)


# =============================================================================
# PART 3: VISUALIZATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ACTIVATION FUNCTION ANALYSIS")
    print("=" * 60)
    
    # =============================================================================
    # Task 2.1: Plot All Activations
    # =============================================================================

    z = np.linspace(-5, 5, 1000)
    
    # TODO: Create subplot grid for activations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    activations = [
        ('Step', step),
        ('Sigmoid', sigmoid),
        ('Tanh', tanh_activation),
        ('ReLU', relu),
        ('Leaky ReLU', leaky_relu)
    ]
    
    for idx, (name, func) in enumerate(activations):
        ax = axes[idx // 3, idx % 3]
        ax.plot(z, func(z), linewidth=2)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    # =============================================================================
    # Task 2.2: Plot Derivatives
    # =============================================================================
    # TODO: Create another 2x3 subplot for derivatives
    # This shows how fast each function changes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    derivatives = [
        ('Sigmoid', sigmoid_derivative, r'$\sigma\'(z) = \sigma(z)(1 - \sigma(z))$'),
        ('Tanh', tanh_derivative, r'$\tanh\'(z) = 1 - \tanh^2(z)$'),
        ('ReLU', relu_derivative, r'$f\'(z) = 1$ if $z > 0$, else $0$'),
        ('Leaky ReLU', leaky_relu_derivative, r'$f\'(z) = 1$ if $z > 0$, else $\alpha$')
    ]

    for idx, (name, func, equation) in enumerate(derivatives):
        ax = axes[idx // 3, idx % 3]
        ax.plot(z, func(z), linewidth=2, color='red')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_title(f'{name} Derivative\n{equation}', fontsize=10)
        ax.set_xlabel('z')
        ax.set_ylabel("f'(z)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # EXPERIMENT A: OUTPUT RANGES
    # =============================================================================
    
    print("\n--- OUTPUT RANGES ---")
    # TODO: Fill in the table
    # | Activation | Min Output | Max Output | Zero-Centered? |
    
    # =============================================================================
    # EXPERIMENT B: GRADIENT VALUES
    # =============================================================================
    
    print("\n--- GRADIENT VALUES ---")
    test_points = [-3, -1, 0, 1, 3]
    
    # TODO: Calculate gradient values at test points
    
    # =============================================================================
    # EXPERIMENT C: VANISHING GRADIENT
    # =============================================================================
    
    print("\n--- VANISHING GRADIENT ---")
    # TODO: What is sigmoid gradient at z = 10? z = -10?
    
    # =============================================================================
    # EXPERIMENT D: DEAD RELU
    # =============================================================================
    
    print("\n--- DEAD RELU ---")
    # TODO: Explain the dying ReLU problem
