"""
Demo 02: Gradient Descent Optimization Comparison

This demo shows trainees how to:
1. Compare SGD, Adam, and RMSprop optimizers
2. Visualize convergence behavior using TensorBoard
3. Understand learning rate sensitivity
4. Experiment with momentum effects

Learning Objectives:
- Compare optimizer behaviors interactively in TensorBoard
- Understand learning rate impact on convergence
- Recognize when to use different optimizers

TensorBoard Visualization:
After running this demo, launch TensorBoard to compare runs:
    tensorboard --logdir=logs/gradient_descent_comparison

References:
- Written Content: 02-gradient-descent-intuition.md
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import shutil

# ============================================================================
# PART 1: 2D Optimization Landscape Visualization
# ============================================================================

print("=" * 70)
print("PART 1: Visualizing Gradient Descent on 2D Surface")
print("=" * 70)

# Define a simple loss surface: Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Create mesh for visualization
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# Gradient descent on Rosenbrock
def gradient_rosenbrock(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return dx, dy

def run_gradient_descent(start, learning_rate, iterations=100):
    """Run vanilla gradient descent"""
    path = [start]
    x, y = start
    
    for _ in range(iterations):
        dx, dy = gradient_rosenbrock(x, y)
        x = x - learning_rate * dx
        y = y - learning_rate * dy
        path.append((x, y))
        
        # Stop if converged
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            break
    
    return np.array(path)

def run_momentum(start, learning_rate, momentum=0.9, iterations=100):
    """Run gradient descent with momentum"""
    path = [start]
    x, y = start
    vx, vy = 0, 0
    
    for _ in range(iterations):
        dx, dy = gradient_rosenbrock(x, y)
        vx = momentum * vx - learning_rate * dx
        vy = momentum * vy - learning_rate * dy
        x = x + vx
        y = y + vy
        path.append((x, y))
    
    return np.array(path)

# Run different optimizers
start = (-1.5, 2.0)
lr = 0.001

path_gd = run_gradient_descent(start, lr, 200)
path_momentum = run_momentum(start, lr, 0.9, 200)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vanilla GD
axes[0].contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
axes[0].plot(path_gd[:, 0], path_gd[:, 1], 'r.-', label='Gradient Descent', linewidth=2, markersize=4)
axes[0].plot(1, 1, 'g*', markersize=15, label='Global Minimum')
axes[0].plot(start[0], start[1], 'ro', markersize=10, label='Start')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Vanilla Gradient Descent')
axes[0].legend()

# With Momentum
axes[1].contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
axes[1].plot(path_momentum[:, 0], path_momentum[:, 1], 'b.-', label='GD + Momentum', linewidth=2, markersize=4)
axes[1].plot(1, 1, 'g*', markersize=15, label='Global Minimum')
axes[1].plot(start[0], start[1], 'bo', markersize=10, label='Start')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Gradient Descent with Momentum')
axes[1].legend()

plt.tight_layout()
plt.savefig('gd_2d_visualization.png', dpi=150)
print("[OK] 2D gradient descent visualization saved")

# ============================================================================
# PART 2: TensorBoard Setup & Comparing Optimizers on MNIST
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Comparing Optimizers on MNIST Classification")
print("=" * 70)

# Setup TensorBoard log directory
LOG_DIR = "logs/gradient_descent_comparison"

# Clear previous logs for fresh comparison
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
    print(f"[INFO] Cleared previous TensorBoard logs at: {LOG_DIR}")

print(f"[INFO] TensorBoard logs will be saved to: {LOG_DIR}")
print("[INFO] Launch TensorBoard with: tensorboard --logdir=logs/gradient_descent_comparison")

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Use subset for faster training
x_train_sub = x_train[:10000]
y_train_sub = y_train[:10000]

def create_model():
    """Create identical model for fair comparison"""
    return keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

# Define optimizers to compare
optimizers = {
    'SGD_lr0.01': keras.optimizers.SGD(learning_rate=0.01),
    'SGD_Momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001)
}

# Train each model with TensorBoard logging
print("\nTraining with different optimizers...")

for name, optimizer in optimizers.items():
    print(f"\nTraining with {name}...")
    
    # Create TensorBoard callback with unique subdirectory for each optimizer
    log_subdir = os.path.join(LOG_DIR, "optimizers", name)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_subdir,
        histogram_freq=1,
        write_graph=True
    )
    
    model = create_model()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        x_train_sub, y_train_sub,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        verbose=0,
        callbacks=[tensorboard_callback]
    )
    
    final_acc = history.history['val_accuracy'][-1]
    print(f"  Final validation accuracy: {final_acc:.4f}")
    print(f"  TensorBoard logs: {log_subdir}")

print("\n[OK] Optimizer comparison logged to TensorBoard")
print("     View with: tensorboard --logdir=logs/gradient_descent_comparison/optimizers")

# ============================================================================
# PART 3: Learning Rate Sensitivity
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Learning Rate Sensitivity")
print("=" * 70)

learning_rates = [0.0001, 0.001, 0.01, 0.1]

print("\nTraining SGD with different learning rates...")

for lr in learning_rates:
    print(f"\n  Learning rate: {lr}")
    
    # Create TensorBoard callback with unique subdirectory for each learning rate
    log_subdir = os.path.join(LOG_DIR, "learning_rates", f"lr_{lr}")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_subdir,
        histogram_freq=1
    )
    
    model = create_model()
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train_sub, y_train_sub,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        verbose=0,
        callbacks=[tensorboard_callback]
    )
    
    final_acc = history.history['val_accuracy'][-1]
    print(f"    Final validation accuracy: {final_acc:.4f}")
    print(f"    TensorBoard logs: {log_subdir}")

print("\n[OK] Learning rate sensitivity logged to TensorBoard")
print("     View with: tensorboard --logdir=logs/gradient_descent_comparison/learning_rates")

print("\nObservations (confirm in TensorBoard):")
print("- LR too small (0.0001): Slow convergence")
print("- LR too large (0.1): Unstable, may diverge")
print("- LR just right (0.01, 0.001): Good convergence")

# ============================================================================
# PART 4: Momentum Effect Visualization
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Effect of Momentum")
print("=" * 70)

momentum_values = [0.0, 0.5, 0.9, 0.99]

print("\nTraining SGD with different momentum values...")

for momentum in momentum_values:
    print(f"\n  Momentum: {momentum}")
    
    # Create TensorBoard callback with unique subdirectory for each momentum value
    log_subdir = os.path.join(LOG_DIR, "momentum", f"momentum_{momentum}")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_subdir,
        histogram_freq=1
    )
    
    model = create_model()
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=momentum),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train_sub, y_train_sub,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        verbose=0,
        callbacks=[tensorboard_callback]
    )
    
    final_acc = history.history['val_accuracy'][-1]
    print(f"    Final validation accuracy: {final_acc:.4f}")
    print(f"    TensorBoard logs: {log_subdir}")

print("\n[OK] Momentum effect logged to TensorBoard")
print("     View with: tensorboard --logdir=logs/gradient_descent_comparison/momentum")

print("\nObservations (confirm in TensorBoard):")
print("- No momentum: Slower convergence")
print("- Momentum 0.9: Faster convergence, smoother loss curve")
print("- Momentum 0.99: May overshoot, needs careful tuning")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Gradient Descent Comparison")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Adam is usually the best default optimizer")
print("2. SGD + Momentum is competitive and can generalize better")
print("3. Learning rate is the most critical hyperparameter")
print("4. Momentum accelerates convergence and smooths updates")
print("5. Use TensorBoard to interactively explore and compare training runs")

print("\nRecommended defaults:")
print("- Adam: learning_rate=0.001")
print("- SGD + Momentum: learning_rate=0.01, momentum=0.9")
print("- RMSprop: learning_rate=0.001")

print("\n" + "=" * 70)
print("TENSORBOARD VISUALIZATION")
print("=" * 70)
print("\nTo view all training comparisons, run:")
print("  tensorboard --logdir=logs/gradient_descent_comparison")
print("\nOr view specific comparisons:")
print("  - Optimizers: tensorboard --logdir=logs/gradient_descent_comparison/optimizers")
print("  - Learning rates: tensorboard --logdir=logs/gradient_descent_comparison/learning_rates")
print("  - Momentum: tensorboard --logdir=logs/gradient_descent_comparison/momentum")
print("\nThen open http://localhost:6006 in your browser")

print("\n" + "=" * 70)