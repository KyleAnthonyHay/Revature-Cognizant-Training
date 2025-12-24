# Exercise 01: Backpropagation by Hand

## Learning Objectives

- Manually compute gradients using the chain rule
- Verify manual calculations with TensorFlow's automatic differentiation
- Visualize gradient flow through network layers
- Understand why vanishing gradients occur with certain activations

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_01_backprop_visualization.py`, you saw gradients flow backward through a network. This exercise has you compute those gradients yourself - first on paper, then verify with code.

## The Math You Need

### Forward Pass Equations

```
# For a 2-layer network:

# Layer 1 (input to hidden)
z1 = W1 * x + b1        # Linear transformation
a1 = sigmoid(z1)        # Activation

# Layer 2 (hidden to output)  
z2 = W2 * a1 + b2       # Linear transformation
y_pred = sigmoid(z2)    # Output activation

# Loss (Binary Cross-Entropy for single sample)
L = -[y * log(y_pred) + (1-y) * log(1-y_pred)]
```

### Backward Pass Equations (Chain Rule)

```
# Gradient of loss w.r.t. output
dL/dy_pred = -y/y_pred + (1-y)/(1-y_pred)

# Gradient through sigmoid: d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z))
dL/dz2 = dL/dy_pred * y_pred * (1 - y_pred)

# Gradients for Layer 2 weights
dL/dW2 = dL/dz2 * a1    # Gradient for W2
dL/db2 = dL/dz2         # Gradient for b2

# Backpropagate to Layer 1
dL/da1 = dL/dz2 * W2
dL/dz1 = dL/da1 * a1 * (1 - a1)

# Gradients for Layer 1 weights
dL/dW1 = dL/dz1 * x     # Gradient for W1
dL/db1 = dL/dz1         # Gradient for b1
```

---

## Part 1: Paper Calculation (20 min)

### Task 1.1: Forward Pass

Given the following network and input:
- Input: x = 0.5
- Target: y = 1
- W1 = 0.8, b1 = 0.2
- W2 = 0.6, b2 = 0.1

Calculate by hand (show your work):

1. **z1** = W1 * x + b1 = ?
2. **a1** = sigmoid(z1) = 1 / (1 + exp(-z1)) = ?
3. **z2** = W2 * a1 + b2 = ?
4. **y_pred** = sigmoid(z2) = ?
5. **Loss** = -[y * log(y_pred) + (1-y) * log(1-y_pred)] = ?

**Hint for sigmoid calculation:**
```
sigmoid(0.6) approximately equals 0.6457
sigmoid(0.7) approximately equals 0.6682
sigmoid(0.5) approximately equals 0.6225
```

### Task 1.2: Backward Pass

Using your forward pass values, compute:

1. **dL/dy_pred** = -y/y_pred + (1-y)/(1-y_pred) = ?
2. **dL/dz2** = dL/dy_pred * y_pred * (1 - y_pred) = ?
3. **dL/dW2** = dL/dz2 * a1 = ?
4. **dL/db2** = dL/dz2 = ?
5. **dL/da1** = dL/dz2 * W2 = ?
6. **dL/dz1** = dL/da1 * a1 * (1 - a1) = ?
7. **dL/dW1** = dL/dz1 * x = ?
8. **dL/db1** = dL/dz1 = ?

---

## Part 2: Code Verification (20 min)

### Task 2.1: Implement Forward Pass

Navigate to `starter_code/backprop_starter.py` and implement:

```python
import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    Formula: sigmoid(z) = 1 / (1 + exp(-z))
    """
    # TODO: Implement sigmoid
    # Hint: Use np.exp() and be careful with overflow
    pass

def forward_pass(x, W1, b1, W2, b2):
    """
    Compute forward pass through 2-layer network.
    
    Returns: (z1, a1, z2, y_pred, all intermediate values for backprop)
    """
    # TODO: Implement forward pass
    # z1 = ...
    # a1 = ...
    # z2 = ...
    # y_pred = ...
    pass

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    Formula: L = -[y * log(y_pred) + (1-y) * log(1-y_pred)]
    """
    # TODO: Implement BCE loss
    # Add small epsilon to prevent log(0)
    pass
```

### Task 2.2: Implement Backward Pass

```python
def backward_pass(x, y, z1, a1, z2, y_pred, W1, b1, W2, b2):
    """
    Compute gradients using chain rule.
    
    Returns: Dictionary with all gradients
    """
    # TODO: Implement backward pass
    # Follow the chain rule equations from Part 1
    
    # dL/dy_pred
    # dL/dz2
    # dL/dW2, dL/db2
    # dL/da1
    # dL/dz1  
    # dL/dW1, dL/db1
    pass
```

### Task 2.3: Verify Against TensorFlow

```python
import tensorflow as tf

def verify_with_tensorflow(x, y, W1, b1, W2, b2):
    """Use TensorFlow's GradientTape to verify our manual gradients."""
    
    # Convert to TensorFlow variables
    W1_tf = tf.Variable([[W1]], dtype=tf.float32)
    b1_tf = tf.Variable([[b1]], dtype=tf.float32)
    W2_tf = tf.Variable([[W2]], dtype=tf.float32)
    b2_tf = tf.Variable([[b2]], dtype=tf.float32)
    x_tf = tf.constant([[x]], dtype=tf.float32)
    y_tf = tf.constant([[y]], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # Forward pass
        z1 = tf.matmul(x_tf, W1_tf) + b1_tf
        a1 = tf.nn.sigmoid(z1)
        z2 = tf.matmul(a1, W2_tf) + b2_tf
        y_pred = tf.nn.sigmoid(z2)
        
        # Loss
        loss = tf.keras.losses.binary_crossentropy(y_tf, y_pred)
    
    # Compute gradients
    gradients = tape.gradient(loss, [W1_tf, b1_tf, W2_tf, b2_tf])
    
    return {
        'dW1': gradients[0].numpy(),
        'db1': gradients[1].numpy(),
        'dW2': gradients[2].numpy(),
        'db2': gradients[3].numpy()
    }

# Compare your manual gradients with TensorFlow's
# They should match within floating-point precision!
```

---

## Part 3: Gradient Visualization (20 min)

### Task 3.1: Gradient Magnitude Analysis

Extend your network to 5 layers and track gradient magnitudes:

```python
def analyze_gradient_flow(x, y, layer_sizes, activation='sigmoid'):
    """
    Build a deep network and analyze gradient magnitudes per layer.
    
    Compare:
    - Sigmoid activation (expect vanishing gradients)
    - ReLU activation (expect stable gradients)
    """
    # TODO: Build network with specified layers
    # TODO: Compute forward pass
    # TODO: Compute backward pass
    # TODO: Record gradient magnitude at each layer
    pass
```

### Task 3.2: Visualization

Create a bar chart showing gradient magnitudes for:
- 5-layer network with Sigmoid
- 5-layer network with ReLU

The Sigmoid network should show diminishing gradients in earlier layers.

---

## Reflection Questions

1. **What values do sigmoid gradients take?** Looking at the formula `sigmoid(z) * (1 - sigmoid(z))`, what is the maximum possible gradient?

2. **Why do gradients vanish with sigmoid?** If each layer's gradient is at most 0.25, what happens after 10 layers?

3. **How does your manual calculation compare to TensorFlow's?** Any discrepancies? Why might tiny differences occur?

4. **In the gradient flow visualization, which layer has the smallest gradients?** Is this problematic for learning?

---

## Definition of Done

- [ ] Paper calculations completed with work shown
- [ ] Forward pass implemented and verified
- [ ] Backward pass implemented and matches TensorFlow
- [ ] Gradient flow visualization created
- [ ] Comparison of Sigmoid vs ReLU gradient flow
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_01_solution.py`. Only check after completing your attempt!

