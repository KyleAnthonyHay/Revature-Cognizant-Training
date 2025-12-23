"""
Exercise 05: Activation Function Analysis
=========================================

SETUP INSTRUCTIONS:
-------------------
1. Create a virtual environment:
   python3 -m venv venv

2. Activate the virtual environment:
   - On macOS/Linux: source venv/bin/activate
   - On Windows: venv\\Scripts\\activate

3. Install required packages:
   pip install -r requirements.txt

4. Run the program:
   python3 ex5-main.py

5. Activation function plots and analysis will display.
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
    z = np.asarray(z)
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
    print(f"{'Activation':<15} {'Min Output':<12} {'Max Output':<12} {'Zero-Centered?':<15}")
    print("-" * 60)

    # Verify by computing actual min/max values
    activations_list = [
        ('Step', step),
        ('Sigmoid', sigmoid),
        ('Tanh', tanh_activation),
        ('ReLU', relu),
        ('Leaky ReLU', leaky_relu)
    ]

    for name, activation_func in activations_list:
        outputs = activation_func(z)
        min_val = outputs.min()
        max_val = outputs.max()
        zero_centered = "Yes" if min_val < 0 else "No"
        print(f"{name:<15} {min_val:<12.4f} {max_val:<12.4f} {zero_centered:<15}")
        
    # =============================================================================
    # EXPERIMENT B: GRADIENT VALUES
    # =============================================================================
    
    print("\n--- GRADIENT VALUES ---")
    test_points = [-3, -1, 0, 1, 3]
    
    # TODO: Calculate gradient values at test points
    print(f"{'z':<8} {'Sigmoid':<15} {'Tanh':<15} {'ReLU':<15} {'Leaky ReLU':<15}")
    print("-" * 70)

    for z_val in test_points:
        sig_grad = sigmoid_derivative(z_val)
        tanh_grad = tanh_derivative(z_val)
        relu_grad = relu_derivative(z_val)
        leaky_grad = leaky_relu_derivative(z_val)
        print(f"{z_val:<8} {sig_grad:<15.6f} {tanh_grad:<15.6f} {relu_grad:<15.6f} {leaky_grad:<15.6f}")
    # =============================================================================
    # EXPERIMENT C: VANISHING GRADIENT
    # =============================================================================
    
    print("\n--- VANISHING GRADIENT ---")
    z_extreme_positive = 10
    z_extreme_negative = -10

    sig_grad_pos = sigmoid_derivative(z_extreme_positive)
    sig_grad_neg = sigmoid_derivative(z_extreme_negative)

    print(f"Sigmoid gradient at z = {z_extreme_positive}: {sig_grad_pos:.8f}")
    print(f"Sigmoid gradient at z = {z_extreme_negative}: {sig_grad_neg:.8f}")
    print(f"\nObservation: At extreme values, gradients are nearly zero.")
    print(f"This causes 'vanishing gradients' - weights barely update during backpropagation.")
    
    print("\nVanishing Gradient Problem:")
    print("Gradients become extremely small (near zero) at extreme input values.")
    print("When gradients are tiny, weight updates during backpropagation are negligible.")
    print("This causes early layers in deep networks to learn very slowly or stop learning.")
    print("Common with sigmoid and tanh activations.")

    
    # =============================================================================
    # EXPERIMENT D: DEAD RELU
    # =============================================================================
    
    print("\n--- DEAD RELU ---")
    print("1. ReLU's gradient when z < 0: 0 (zero gradient)")
    print()
    print("2. Dying ReLU problem:")
    print("   When z < 0, ReLU outputs 0 and has gradient 0.")
    print("   If weights cause z to stay negative, the neuron outputs 0 forever.")
    print("   With gradient = 0, weights don't update, so the neuron stays \"dead\".")
    print("   This reduces network capacity and learning ability.")
    print()
    print("3. Leaky ReLU solution:")
    print("   Uses a small positive gradient (alpha, typically 0.01) for z < 0.")
    print("   This allows weights to update even when z is negative.")
    print("   Prevents neurons from dying completely.")
