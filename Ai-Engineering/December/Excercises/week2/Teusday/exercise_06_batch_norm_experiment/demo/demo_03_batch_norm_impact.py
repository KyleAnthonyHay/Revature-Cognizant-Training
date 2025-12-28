"""
Demo 03: Batch Normalization Impact

This demo shows trainees how to:
1. Understand internal covariate shift problem
2. Add batch normalization to networks
3. Compare training with/without batch normalization
4. Visualize the benefits: faster training, higher learning rates

Learning Objectives:
- Understand why batch normalization helps training
- Learn to add BatchNormalization layers
- See the convergence speed improvement in TensorBoard

TensorBoard Visualization:
After running this demo, launch TensorBoard to compare runs:
    tensorboard --logdir=logs/batch_norm

References:
- Written Content: 04-batch-normalization.md
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import shutil

# ============================================================================
# PART 1: Understanding Internal Covariate Shift
# ============================================================================

print("=" * 70)
print("PART 1: Understanding Internal Covariate Shift")
print("=" * 70)

print("\nInternal Covariate Shift Problem:")
print("-" * 40)
print("During training, the distribution of layer inputs changes")
print("because previous layer weights are constantly being updated.")
print("\nThis forces each layer to continuously adapt to new distributions,")
print("slowing down training and requiring smaller learning rates.")
print("\nBatch Normalization Solution:")
print("Normalize layer inputs to have mean=0, variance=1")
print("This stabilizes the distribution across training iterations.")

# Demonstrate activation statistics without batch norm
print("\n" + "-" * 40)
print("Visualizing activation statistics across layers...")

# Simple model without batch norm
model_no_bn = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Create data
X_sample = np.random.randn(1000, 784).astype('float32')

# Compute activation statistics by passing data through the model
print("\nActivation statistics (before training, no batch norm):")

# Get activations layer by layer
current_output = X_sample
for i, layer in enumerate(model_no_bn.layers[:-1]):
    current_output = layer(current_output)
    mean = np.mean(current_output)
    std = np.std(current_output)
    print(f"Layer {i+1}: mean={mean:.4f}, std={std:.4f}")

# ============================================================================
# PART 2: Building Models With and Without Batch Normalization
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Building Models With and Without Batch Norm")
print("=" * 70)

def create_model_without_bn():
    """Model without batch normalization"""
    return keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name='no_batch_norm')

def create_model_with_bn():
    """Model with batch normalization"""
    return keras.Sequential([
        layers.Dense(256, use_bias=False, input_shape=(784,)),  # No bias when using BN
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(32, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(10, activation='softmax')
    ], name='with_batch_norm')

# Display architectures
model_no_bn = create_model_without_bn()
model_with_bn = create_model_with_bn()

print("\nModel WITHOUT Batch Normalization:")
print(f"Total parameters: {model_no_bn.count_params():,}")

print("\nModel WITH Batch Normalization:")
print(f"Total parameters: {model_with_bn.count_params():,}")
print("(Extra parameters are gamma and beta for each BN layer)")

# ============================================================================
# PART 3: Training Comparison
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Training Comparison")
print("=" * 70)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Use subset for faster demo
x_train_sub = x_train[:20000]
y_train_sub = y_train[:20000]

# Train without batch norm
print("\nTraining WITHOUT batch normalization...")
model_no_bn = create_model_without_bn()
model_no_bn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TensorBoard for no BN model
log_dir_no_bn = "logs/batch_norm/no_bn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir_no_bn, exist_ok=True)
tb_no_bn = keras.callbacks.TensorBoard(log_dir=log_dir_no_bn, histogram_freq=1)

start_time = time.time()
history_no_bn = model_no_bn.fit(
    x_train_sub, y_train_sub,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tb_no_bn],
    verbose=0
)
time_no_bn = time.time() - start_time
print(f"Training time: {time_no_bn:.2f}s")
print(f"Final val accuracy: {history_no_bn.history['val_accuracy'][-1]:.4f}")

# Train with batch norm
print("\nTraining WITH batch normalization...")
model_with_bn = create_model_with_bn()
model_with_bn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TensorBoard for with BN model
log_dir_with_bn = "logs/batch_norm/with_bn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir_with_bn, exist_ok=True)
tb_with_bn = keras.callbacks.TensorBoard(log_dir=log_dir_with_bn, histogram_freq=1)

start_time = time.time()
history_with_bn = model_with_bn.fit(
    x_train_sub, y_train_sub,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tb_with_bn],
    verbose=0
)
time_with_bn = time.time() - start_time
print(f"Training time: {time_with_bn:.2f}s")
print(f"Final val accuracy: {history_with_bn.history['val_accuracy'][-1]:.4f}")

# ============================================================================
# PART 4: View Training Comparison (TensorBoard)
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Visualizing Training Comparison")
print("=" * 70)

print("\nTraining metrics are now logged to TensorBoard!")
print(f"To compare with/without batch normalization, run:")
print(f"  tensorboard --logdir=logs/batch_norm")
print(f"  Then navigate to http://localhost:6006")
print("\nIn TensorBoard:")
print("  - Use the 'Runs' selector to toggle between no_bn_* and with_bn_*")
print("  - Compare loss and accuracy curves side by side")
print("  - Notice faster convergence with batch normalization")

# ============================================================================
# PART 5: Higher Learning Rates with Batch Norm (TensorBoard)
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Higher Learning Rates with Batch Normalization")
print("=" * 70)

print("\nWithout batch norm, high learning rates cause instability.")
print("With batch norm, we can use much higher learning rates!")

# Setup TensorBoard log directory for learning rate experiments
LR_LOG_DIR = "logs/batch_norm/learning_rate_comparison"

# Clear previous logs for fresh comparison
if os.path.exists(LR_LOG_DIR):
    shutil.rmtree(LR_LOG_DIR)
    print(f"\n[INFO] Cleared previous TensorBoard logs at: {LR_LOG_DIR}")

print(f"[INFO] TensorBoard logs will be saved to: {LR_LOG_DIR}")

learning_rates = [0.001, 0.01, 0.05, 0.1]

# Test without BN
print("\nTesting learning rates WITHOUT batch norm:")
for lr in learning_rates:
    print(f"\n  Training with LR={lr}...")
    
    # Create TensorBoard callback with unique subdirectory
    log_subdir = os.path.join(LR_LOG_DIR, f"no_bn_lr_{lr}")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_subdir,
        histogram_freq=1
    )
    
    model = create_model_without_bn()
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train_sub, y_train_sub,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=0,
        callbacks=[tensorboard_callback]
    )
    final_acc = history.history['val_accuracy'][-1]
    print(f"    Final val_acc={final_acc:.4f}")
    print(f"    TensorBoard logs: {log_subdir}")

# Test with BN
print("\nTesting learning rates WITH batch norm:")
for lr in learning_rates:
    print(f"\n  Training with LR={lr}...")
    
    # Create TensorBoard callback with unique subdirectory
    log_subdir = os.path.join(LR_LOG_DIR, f"with_bn_lr_{lr}")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_subdir,
        histogram_freq=1
    )
    
    model = create_model_with_bn()
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train_sub, y_train_sub,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=0,
        callbacks=[tensorboard_callback]
    )
    final_acc = history.history['val_accuracy'][-1]
    print(f"    Final val_acc={final_acc:.4f}")
    print(f"    TensorBoard logs: {log_subdir}")

print("\n[OK] Learning rate comparison logged to TensorBoard")
print("     View with: tensorboard --logdir=logs/batch_norm/learning_rate_comparison")
print("\nIn TensorBoard, use the regex filter to compare:")
print("  - All no_bn runs: no_bn.*")
print("  - All with_bn runs: with_bn.*")
print("  - Same LR, different BN: .*lr_0.01")

print("\nObservation (confirm in TensorBoard):")
print("- Without BN: High LRs (0.05, 0.1) cause training to fail")
print("- With BN: Network tolerates higher learning rates")
print("- This allows faster convergence!")

# ============================================================================
# PART 6: Examining Batch Normalization Parameters
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Batch Normalization Parameters")
print("=" * 70)

# Get a trained BN layer's parameters
model_with_bn = create_model_with_bn()
model_with_bn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model_with_bn.fit(x_train_sub[:1000], y_train_sub[:1000], epochs=5, verbose=0)

print("\nBatch Normalization layer parameters:")
for layer in model_with_bn.layers:
    if isinstance(layer, layers.BatchNormalization):
        weights = layer.get_weights()
        gamma, beta, moving_mean, moving_var = weights
        
        print(f"\n{layer.name}:")
        print(f"  gamma (scale):     mean={gamma.mean():.4f}, std={gamma.std():.4f}")
        print(f"  beta (shift):      mean={beta.mean():.4f}, std={beta.std():.4f}")
        print(f"  moving_mean:       mean={moving_mean.mean():.4f}, std={moving_mean.std():.4f}")
        print(f"  moving_variance:   mean={moving_var.mean():.4f}, std={moving_var.std():.4f}")
        break

print("\nExplanation:")
print("- gamma: Learned scale parameter")
print("- beta: Learned shift parameter")
print("- moving_mean: Running average of batch means (for inference)")
print("- moving_variance: Running average of batch variances (for inference)")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Batch Normalization Impact")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Batch norm normalizes layer inputs (mean=0, var=1)")
print("2. Solves internal covariate shift problem")
print("3. Enables higher learning rates -> faster training")
print("4. Acts as mild regularizer")
print("5. Has trainable params: gamma (scale) and beta (shift)")
print("6. Uses moving averages for inference (not batch stats)")

print("\nBest Practices:")
print("- Place BN after Dense, before activation")
print("- Set use_bias=False in Dense when using BN")
print("- Works best with batch size >= 32")

print("\n" + "=" * 70)
print("TENSORBOARD VISUALIZATION")
print("=" * 70)
print("\nTo view all batch normalization experiments, run:")
print("  tensorboard --logdir=logs/batch_norm")
print("\nOr view specific comparisons:")
print("  - With/without BN: tensorboard --logdir=logs/batch_norm")
print("  - LR comparison: tensorboard --logdir=logs/batch_norm/learning_rate_comparison")
print("\nThen open http://localhost:6006 in your browser")

print("\n" + "=" * 70)
