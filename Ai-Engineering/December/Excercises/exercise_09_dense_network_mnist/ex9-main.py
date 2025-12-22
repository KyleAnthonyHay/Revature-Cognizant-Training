"""
Exercise 09: Dense Network for MNIST
====================================

Build and train a Dense neural network for digit classification.
Target: > 95% test accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("MNIST DIGIT CLASSIFICATION WITH DENSE NETWORK")
print("=" * 60)

# =============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# =============================================================================

print("\n--- PART 1: DATA LOADING AND EXPLORATION ---")

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Explore the data
print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Label range: {y_train.min()} to {y_train.max()}")
print(f"Pixel value range: {X_train.min()} to {X_train.max()}")

# Visualize some samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# =============================================================================
# PART 2: DATA PREPROCESSING
# =============================================================================

print("\n--- PART 2: DATA PREPROCESSING ---")

# Task 2.1: Normalize Pixel Values to [0, 1] range
print("\nNormalizing pixel values...")
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

# Task 2.2: Flatten Images from (n, 28, 28) to (n, 784)
print("Flattening images...")
X_train_flat = X_train_norm.reshape(-1, 784)
X_test_flat = X_test_norm.reshape(-1, 784)

print(f"Flattened training shape: {X_train_flat.shape}")
print(f"Flattened test shape: {X_test_flat.shape}")

# Task 2.3: Verify Preprocessing
print("\nVerifying preprocessing...")
assert X_train_flat.shape == (60000, 784), f"Expected (60000, 784), got {X_train_flat.shape}"
assert X_test_flat.shape == (10000, 784), f"Expected (10000, 784), got {X_test_flat.shape}"
assert X_train_flat.max() <= 1.0, f"Expected max <= 1.0, got {X_train_flat.max()}"
assert X_train_flat.min() >= 0.0, f"Expected min >= 0.0, got {X_train_flat.min()}"
print("Preprocessing checks passed!")

# =============================================================================
# PART 3: BUILD THE MODEL
# =============================================================================

print("\n--- PART 3: BUILD THE MODEL ---")

# Task 3.1: Design Your Architecture
# Architecture: 784 -> 128 -> 64 -> 10
# Input: 784 features (flattened 28x28 image)
# Hidden layers: ReLU activation
# Output: 10 classes (digits 0-9) with Softmax

print("\nDesigning architecture...")
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
], name='mnist_dense')

# Task 3.2: Calculate Expected Parameters
# Layer 1: 784 * 128 + 128 = 100,480
# Layer 2: 128 * 64 + 64 = 8,256
# Output: 64 * 10 + 10 = 650
# Total: ~109,386 parameters

print("\nModel Summary:")
model.summary()

# Task 3.3: Compile the Model
print("\nCompiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled successfully!")

# =============================================================================
# PART 4: TRAIN THE MODEL
# =============================================================================

print("\n--- PART 4: TRAIN THE MODEL ---")

# Task 4.1: Initial Training
print("\nStarting training...")
history = model.fit(
    X_train_flat, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Task 4.2: Plot Training History
print("\nPlotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history.history['loss'], label='Training')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Loss Over Time')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(history.history['accuracy'], label='Training')
axes[1].plot(history.history['val_accuracy'], label='Validation')
axes[1].set_title('Accuracy Over Time')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Task 4.3: Analyze the Curves
# Questions to consider:
# - Is the model overfitting? (Check if training accuracy >> validation accuracy)
# - Did validation accuracy plateau? (Check if val_accuracy stopped improving)
# - Should you train for more epochs? (Check if loss is still decreasing)

print("\nTraining completed!")

# =============================================================================
# PART 5: EVALUATE
# =============================================================================

print("\n--- PART 5: EVALUATE ---")

# Task 5.1: Test Set Performance
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")

# Check if target accuracy achieved
if test_acc >= 0.95:
    print("SUCCESS! Target accuracy achieved!")
else:
    print(f"Keep trying! Need {(0.95 - test_acc):.2%} more accuracy")

# Task 5.2: Confusion Analysis
print("\nAnalyzing misclassified examples...")
y_pred = model.predict(X_test_flat, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Find misclassified examples
misclassified = np.where(y_pred_classes != y_test)[0]
print(f"Misclassified: {len(misclassified)} / {len(y_test)}")

# Visualize some misclassified examples
if len(misclassified) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, idx in enumerate(misclassified[:10]):
        ax = axes[i // 5, i % 5]
        ax.imshow(X_test[idx], cmap='gray')
        ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
        ax.axis('off')
    plt.suptitle("Misclassified Examples")
    plt.tight_layout()
    plt.show()
else:
    print("No misclassified examples found!")
