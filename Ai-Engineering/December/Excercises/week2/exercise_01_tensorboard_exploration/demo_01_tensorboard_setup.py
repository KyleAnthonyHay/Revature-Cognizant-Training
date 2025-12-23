"""
Demo 01: TensorBoard Setup and Real-Time Metric Visualization

This demo shows trainees how to:
1. Install and configure TensorBoard
2. Set up the TensorBoard callback in Keras
3. Visualize training metrics in real-time
4. Compare multiple experiment runs

Learning Objectives:
- Understand TensorBoard's role in model development
- Learn to log scalars, histograms, and graphs
- Master the TensorBoard interface for analysis

References:
- Written Content: 01-tensorboard-visualization.md
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime
import os

# ============================================================================
# PART 1: Basic TensorBoard Setup
# ============================================================================

print("=" * 70)
print("PART 1: Setting Up TensorBoard for MNIST Classification")
print("=" * 70)

# Load MNIST dataset
print("\n[Step 1] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# Build a simple model
print("\n[Step 2] Building neural network...")
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,), name='hidden_1'),
    layers.Dense(64, activation='relu', name='hidden_2'),
    layers.Dense(10, activation='softmax', name='output')
], name='mnist_classifier')

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================================
# PART 2: Configure TensorBoard Callback
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Configuring TensorBoard Callback")
print("=" * 70)

# Create unique log directory with timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

print(f"\n[Step 3] TensorBoard logs will be saved to: {log_dir}")

# Configure TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,        # Log weight histograms every epoch
    write_graph=True,        # Visualize the model graph
    write_images=False,      # Don't save model weights as images (saves space)
    update_freq='epoch',     # Log metrics after each epoch
    profile_batch=0          # Disable profiling (can be enabled for performance analysis)
)

print("\nTensorBoard Callback Configuration:")
print(f"  - Log directory: {log_dir}")
print(f"  - Histogram logging: Every epoch")
print(f"  - Graph visualization: Enabled")
print(f"  - Update frequency: Per epoch")

# ============================================================================
# PART 3: Train Model with TensorBoard Logging
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Training Model with TensorBoard")
print("=" * 70)

print("\n[Step 4] Starting training...")
print("INSTRUCTOR NOTE: Open another terminal and run:")
print(f"  tensorboard --logdir={log_dir}")
print("  Then navigate to http://localhost:6006\n")

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tensorboard_callback],
    verbose=1
)

# ============================================================================
# PART 4: Multiple Experiments for Comparison
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Running Multiple Experiments")
print("=" * 70)

print("\nNow let's run experiments with different configurations...")
print("This demonstrates TensorBoard's ability to compare runs.\n")

# Experiment 1: Baseline (already trained above)
# Experiment 2: With Dropout
print("[Experiment 2] Training with Dropout...")

log_dir_dropout = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_dropout"
os.makedirs(log_dir_dropout, exist_ok=True)

model_dropout = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
], name='mnist_classifier_dropout')

model_dropout.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

tensorboard_dropout = keras.callbacks.TensorBoard(
    log_dir=log_dir_dropout,
    histogram_freq=1,
    write_graph=True
)

history_dropout = model_dropout.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tensorboard_dropout],
    verbose=0
)

print("[OK] Dropout experiment complete")

# Experiment 3: Larger Network
print("\n[Experiment 3] Training larger network...")

log_dir_large = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_large"
os.makedirs(log_dir_large, exist_ok=True)

model_large = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
], name='mnist_classifier_large')

model_large.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

tensorboard_large = keras.callbacks.TensorBoard(
    log_dir=log_dir_large,
    histogram_freq=1,
    write_graph=True
)

history_large = model_large.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tensorboard_large],
    verbose=0
)

print("[OK] Large network experiment complete")

# ============================================================================
# PART 5: Analyzing Results in TensorBoard
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: TensorBoard Analysis Guide")
print("=" * 70)

print("\nTo view all experiments in TensorBoard:")
print(f"  tensorboard --logdir=logs/fit")
print(f"  Navigate to http://localhost:6006\n")

print("What to explore in TensorBoard:")
print("\n1. SCALARS Tab:")
print("   - Compare training/validation loss across all 3 experiments")
print("   - Compare accuracy curves")
print("   - Notice: Dropout model may have higher training loss but better validation")
print("   - Notice: Larger model may overfit (train/val gap)")

print("\n3. DISTRIBUTIONS Tab:")
print("   - Watch weight distributions evolve during training")
print("   - Notice: Healthy training shows weights spreading out")
print("   - Notice: Dead neurons show weights stuck near zero")

print("\n4. HISTOGRAMS Tab:")
print("   - 3D view of weight distributions over time")
print("   - See how each layer's weights change epoch-by-epoch")

# ============================================================================
# PART 6: Custom Scalar Logging (Advanced)
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Custom Scalar Logging")
print("=" * 70)

print("\nDemonstrating custom metric logging...")

log_dir_custom = "logs/custom/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir_custom, exist_ok=True)

# Create a custom training loop to log custom metrics
file_writer = tf.summary.create_file_writer(log_dir_custom)

print(f"Custom logs will be saved to: {log_dir_custom}")

# Simulate logging custom metrics
with file_writer.as_default():
    for epoch in range(10):
        # Simulate custom metrics
        learning_rate = 0.001 * (0.95 ** epoch)  # Decaying learning rate
        gradient_norm = np.random.uniform(0.1, 0.5)  # Simulated gradient norm
        
        # Log custom scalars
        tf.summary.scalar('learning_rate', learning_rate, step=epoch)
        tf.summary.scalar('gradient_norm', gradient_norm, step=epoch)
        
        file_writer.flush()

print("[OK] Custom metrics logged")
print(f"\nView custom metrics:")
print(f"  tensorboard --logdir={log_dir_custom}")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: TensorBoard Setup")
print("=" * 70)

print("\nKey Takeaways:")
print("1. TensorBoard visualizes training in real-time")
print("2. Use unique log directories for each experiment")
print("3. histogram_freq=1 logs weight distributions")
print("4. Compare multiple runs by pointing --logdir to parent folder")
print("5. Custom scalars enable tracking any metric")

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)
print("\nShow trainees:")
print("1. Open TensorBoard in browser while training is running")
print("2. Refresh to see real-time updates")
print("3. Use the 'Runs' selector to compare experiments")
print("4. Zoom and pan on loss curves")
print("5. Download charts as PNG/SVG for reports")

print("\nCommon Issues:")
print("- Port 6006 already in use -> kill existing TensorBoard process")
print("- Logs not appearing -> check log_dir path is correct")
print("- Blank graphs -> wait for first epoch to complete")

print("\n" + "=" * 70)

