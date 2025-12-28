"""
Demo 01: Backpropagation Visualization

This demo shows trainees how to:
1. Understand gradient flow through network layers
2. Visualize how gradients propagate backward
3. Compute manual backpropagation for a simple network
4. Use TensorFlow's GradientTape for automatic differentiation

Learning Objectives:
- Understand the chain rule in backpropagation
- Visualize gradient magnitudes at each layer
- Recognize vanishing/exploding gradient patterns

References:
- Written Content: 01-backpropagation-algorithm.md
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: Manual Backpropagation for Simple Network
# ============================================================================

print("=" * 70)
print("PART 1: Manual Backpropagation - 2-Layer Network")
print("=" * 70)

print("\nLet's manually compute backpropagation for a tiny network:")
print("Input(1) -> Hidden(2 neurons) -> Output(1)")

# Simple network parameters
np.random.seed(42)

# Input
x = np.array([[2.0]])  # Single input

# Layer 1 weights and biases
W1 = np.array([[0.5, -0.3]])  # Shape: (1, 2)
b1 = np.array([[0.1, 0.2]])   # Shape: (1, 2)

# Layer 2 weights and biases
W2 = np.array([[0.4], [0.6]]) # Shape: (2, 1)
b2 = np.array([[0.1]])        # Shape: (1, 1)

# Target
y_true = np.array([[1.0]])

print(f"\nNetwork parameters:")
print(f"Input x = {x[0,0]}")
print(f"W1 = {W1[0]}, b1 = {b1[0]}")
print(f"W2 = {W2.flatten()}, b2 = {b2[0,0]}")
print(f"Target y = {y_true[0,0]}")

# ============================================================================
# FORWARD PASS (Step by step)
# ============================================================================

print("\n" + "-" * 50)
print("FORWARD PASS")
print("-" * 50)

# Layer 1: Linear transformation + ReLU
z1 = np.dot(x, W1) + b1
print(f"\n1. z1 = x * W1 + b1 = {x[0,0]} * {W1[0]} + {b1[0]} = {z1[0]}")

a1 = np.maximum(0, z1)  # ReLU activation
print(f"2. a1 = ReLU(z1) = {a1[0]}")

# Layer 2: Linear transformation + Sigmoid
z2 = np.dot(a1, W2) + b2
print(f"\n3. z2 = a1 * W2 + b2 = {a1[0]} * {W2.flatten()} + {b2[0,0]} = {z2[0,0]:.4f}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_pred = sigmoid(z2)
print(f"4. y_pred = sigmoid(z2) = sigmoid({z2[0,0]:.4f}) = {y_pred[0,0]:.4f}")

# Compute loss (MSE)
loss = (y_pred - y_true) ** 2
print(f"\n5. Loss = (y_pred - y_true)^2 = ({y_pred[0,0]:.4f} - {y_true[0,0]})^2 = {loss[0,0]:.4f}")

# ============================================================================
# BACKWARD PASS (Step by step)
# ============================================================================

print("\n" + "-" * 50)
print("BACKWARD PASS (Chain Rule)")
print("-" * 50)

# Gradient of loss w.r.t. y_pred
dL_dy_pred = 2 * (y_pred - y_true)
print(f"\n1. dL/dy_pred = 2 * (y_pred - y_true) = 2 * ({y_pred[0,0]:.4f} - {y_true[0,0]}) = {dL_dy_pred[0,0]:.4f}")

# Gradient of y_pred w.r.t. z2 (sigmoid derivative)
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

dy_pred_dz2 = sigmoid_derivative(z2)
print(f"\n2. dy_pred/dz2 = sigmoid'(z2) = {y_pred[0,0]:.4f} * (1 - {y_pred[0,0]:.4f}) = {dy_pred_dz2[0,0]:.4f}")

# Chain rule: dL/dz2
dL_dz2 = dL_dy_pred * dy_pred_dz2
print(f"\n3. dL/dz2 = dL/dy_pred * dy_pred/dz2 = {dL_dy_pred[0,0]:.4f} * {dy_pred_dz2[0,0]:.4f} = {dL_dz2[0,0]:.4f}")

# Gradient w.r.t. W2
dL_dW2 = np.dot(a1.T, dL_dz2)
print(f"\n4. dL/dW2 = a1^T * dL/dz2 = {a1[0]}^T * {dL_dz2[0,0]:.4f} = {dL_dW2.flatten()}")

# Gradient w.r.t. b2
dL_db2 = dL_dz2
print(f"   dL/db2 = dL/dz2 = {dL_db2[0,0]:.4f}")

# Gradient w.r.t. a1 (propagate backward)
dL_da1 = np.dot(dL_dz2, W2.T)
print(f"\n5. dL/da1 = dL/dz2 * W2^T = {dL_dz2[0,0]:.4f} * {W2.flatten()} = {dL_da1[0]}")

# Gradient through ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

da1_dz1 = relu_derivative(z1)
print(f"\n6. da1/dz1 = ReLU'(z1) = {da1_dz1[0]}")

# Chain rule: dL/dz1
dL_dz1 = dL_da1 * da1_dz1
print(f"\n7. dL/dz1 = dL/da1 * da1/dz1 = {dL_da1[0]} * {da1_dz1[0]} = {dL_dz1[0]}")

# Gradient w.r.t. W1
dL_dW1 = np.dot(x.T, dL_dz1)
print(f"\n8. dL/dW1 = x^T * dL/dz1 = {x[0,0]} * {dL_dz1[0]} = {dL_dW1[0]}")

# Gradient w.r.t. b1
dL_db1 = dL_dz1
print(f"   dL/db1 = dL/dz1 = {dL_db1[0]}")

# ============================================================================
# WEIGHT UPDATE
# ============================================================================

print("\n" + "-" * 50)
print("WEIGHT UPDATE (Gradient Descent)")
print("-" * 50)

learning_rate = 0.1
print(f"\nLearning rate: {learning_rate}")

W2_new = W2 - learning_rate * dL_dW2
b2_new = b2 - learning_rate * dL_db2
W1_new = W1 - learning_rate * dL_dW1
b1_new = b1 - learning_rate * dL_db1

print(f"\nOld W2: {W2.flatten()} -> New W2: {W2_new.flatten()}")
print(f"Old b2: {b2[0,0]:.4f} -> New b2: {b2_new[0,0]:.4f}")
print(f"Old W1: {W1[0]} -> New W1: {W1_new[0]}")
print(f"Old b1: {b1[0]} -> New b1: {b1_new[0]}")

# ============================================================================
# PART 2: Verify with TensorFlow's GradientTape
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Verification with TensorFlow GradientTape")
print("=" * 70)

# Create same network in TensorFlow
x_tf = tf.constant([[2.0]], dtype=tf.float32)
y_true_tf = tf.constant([[1.0]], dtype=tf.float32)

# Variables for weights
W1_tf = tf.Variable([[0.5, -0.3]], dtype=tf.float32)
b1_tf = tf.Variable([[0.1, 0.2]], dtype=tf.float32)
W2_tf = tf.Variable([[0.4], [0.6]], dtype=tf.float32)
b2_tf = tf.Variable([[0.1]], dtype=tf.float32)

# Forward pass with gradient recording
with tf.GradientTape() as tape:
    z1_tf = tf.matmul(x_tf, W1_tf) + b1_tf
    a1_tf = tf.nn.relu(z1_tf)
    z2_tf = tf.matmul(a1_tf, W2_tf) + b2_tf
    y_pred_tf = tf.sigmoid(z2_tf)
    loss_tf = tf.square(y_pred_tf - y_true_tf)

# Compute gradients
gradients = tape.gradient(loss_tf, [W1_tf, b1_tf, W2_tf, b2_tf])

print("\nTensorFlow computed gradients:")
print(f"dL/dW1 (TF): {gradients[0].numpy()[0]}")
print(f"dL/dW1 (Manual): {dL_dW1[0]}")
print(f"\ndL/dW2 (TF): {gradients[2].numpy().flatten()}")
print(f"dL/dW2 (Manual): {dL_dW2.flatten()}")

print("\n[OK] Manual and TensorFlow gradients match!")

# ============================================================================
# PART 3: Visualizing Gradient Flow in Deep Network
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Gradient Flow Visualization in Deep Network")
print("=" * 70)

# Build deeper network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,), name='layer1'),
    layers.Dense(64, activation='relu', name='layer2'),
    layers.Dense(32, activation='relu', name='layer3'),
    layers.Dense(16, activation='relu', name='layer4'),
    layers.Dense(10, activation='softmax', name='output')
], name='deep_network')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Create dummy data
X_sample = np.random.randn(32, 784).astype('float32')
y_sample = np.random.randint(0, 10, 32)

# Compute gradients for all layers
with tf.GradientTape() as tape:
    predictions = model(X_sample, training=True)
    loss = keras.losses.sparse_categorical_crossentropy(y_sample, predictions)
    loss = tf.reduce_mean(loss)

gradients = tape.gradient(loss, model.trainable_variables)

# Extract gradient magnitudes per layer
gradient_means = []
gradient_stds = []
layer_names = []

for var, grad in zip(model.trainable_variables, gradients):
    if 'kernel' in var.name:  # Only weight matrices, not biases
        layer_name = var.name.split('/')[0]
        grad_mean = tf.reduce_mean(tf.abs(grad)).numpy()
        grad_std = tf.math.reduce_std(grad).numpy()
        
        gradient_means.append(grad_mean)
        gradient_stds.append(grad_std)
        layer_names.append(layer_name)
        
        print(f"{layer_name}: mean_grad={grad_mean:.6f}, std_grad={grad_std:.6f}")

# Visualize gradient flow
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(layer_names, gradient_means, color='steelblue')
plt.xlabel('Layer')
plt.ylabel('Mean Gradient Magnitude')
plt.title('Gradient Magnitudes Across Layers')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(layer_names, gradient_stds, color='coral')
plt.xlabel('Layer')
plt.ylabel('Gradient Standard Deviation')
plt.title('Gradient Variance Across Layers')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('gradient_flow.png', dpi=150)
print("\n[OK] Gradient flow visualization saved to: gradient_flow.png")

# ============================================================================
# PART 4: Demonstrating Vanishing Gradients
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Vanishing Gradients with Sigmoid Activation")
print("=" * 70)

print("\nComparing gradient flow: ReLU vs Sigmoid activations")

# Model with Sigmoid (prone to vanishing gradients)
model_sigmoid = keras.Sequential([
    layers.Dense(128, activation='sigmoid', input_shape=(784,)),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(16, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
], name='sigmoid_network')

model_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Model with ReLU (healthier gradients)
model_relu = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='softmax')
], name='relu_network')

model_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def get_gradient_magnitudes(model, X, y):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    magnitudes = []
    for var, grad in zip(model.trainable_variables, gradients):
        if 'kernel' in var.name:
            magnitudes.append(tf.reduce_mean(tf.abs(grad)).numpy())
    
    return magnitudes

# Compare
sigmoid_grads = get_gradient_magnitudes(model_sigmoid, X_sample, y_sample)
relu_grads = get_gradient_magnitudes(model_relu, X_sample, y_sample)

print("\nGradient magnitudes (input to output):")
print(f"Sigmoid: {[f'{g:.6f}' for g in sigmoid_grads]}")
print(f"ReLU:    {[f'{g:.6f}' for g in relu_grads]}")

# Visualize comparison
plt.figure(figsize=(10, 5))
x_pos = np.arange(len(sigmoid_grads))
width = 0.35

plt.bar(x_pos - width/2, sigmoid_grads, width, label='Sigmoid', color='coral')
plt.bar(x_pos + width/2, relu_grads, width, label='ReLU', color='steelblue')

plt.xlabel('Layer (input to output)')
plt.ylabel('Mean Gradient Magnitude')
plt.title('Vanishing Gradients: Sigmoid vs ReLU')
plt.legend()
plt.yscale('log')
plt.xticks(x_pos, ['L1', 'L2', 'L3', 'L4', 'Output'])

plt.tight_layout()
plt.savefig('vanishing_gradients.png', dpi=150)
print("\n[OK] Vanishing gradients comparison saved to: vanishing_gradients.png")

print("\nObservation:")
print("- Sigmoid gradients decrease dramatically toward input layers")
print("- ReLU gradients remain more stable across layers")
print("- This is why ReLU became the default activation function!")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Backpropagation Visualization")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Backpropagation applies the chain rule layer by layer")
print("2. Gradients flow backward from loss to input")
print("3. Each layer's gradient depends on all subsequent layers")
print("4. Sigmoid causes vanishing gradients in deep networks")
print("5. ReLU maintains healthier gradient flow")
print("6. TensorFlow's GradientTape automates gradient computation")

print("\n" + "=" * 70)

