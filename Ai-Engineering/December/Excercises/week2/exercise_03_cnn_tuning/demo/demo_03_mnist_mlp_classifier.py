"""
Demo 03: MNIST Digit Classification with Convolutional Neural Network

This demo builds on Week 1's CNN concepts by implementing a complete CNN classifier.
Trainees apply what they learned about convolutions to a working model.

Learning Objectives:
- Apply CNN concepts from Week 1 to a working implementation
- Use proper layers.Flatten() instead of manual reshaping
- Compare CNN efficiency vs MLP (fewer parameters, better accuracy)
- Monitor training with TensorBoard and early stopping

References:
- Week 1: CNN Fundamentals
- Written Content: 03-cnn-digit-classifier.md
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime
import os

os.makedirs('logs/mnist_cnn', exist_ok=True)

# ============================================================================
# PART 1: Load and Explore MNIST Dataset
# ============================================================================

print("=" * 70)
print("PART 1: Loading MNIST Dataset")
print("=" * 70)

# Load MNIST
print("\n[Step 1] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Dataset loaded:")
print(f"  Training samples: {x_train.shape[0]}")
print(f"  Test samples: {x_test.shape[0]}")
print(f"  Image dimensions: {x_train.shape[1]} x {x_train.shape[2]}")

# Visualize sample digits
print("\n[Step 2] Visualizing sample digits...")
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(20):
    ax = axes[i // 10, i % 10]
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(f'{y_train[i]}', fontsize=12)
    ax.axis('off')

plt.suptitle('MNIST Training Samples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150)
print("[OK] Sample visualization saved to: mnist_samples.png")

# ============================================================================
# PART 2: Preprocess Data for CNN
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Data Preprocessing for CNN")
print("=" * 70)

# Reshape to add channel dimension: (28, 28) -> (28, 28, 1)
print("\n[Step 1] Adding channel dimension for CNN...")
print(f"Original shape: {x_train.shape}")
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(f"CNN shape:      {x_train.shape}")
print("  -> (samples, height, width, channels)")
print("  -> 1 channel = grayscale")

# Normalize pixel values
print("\n[Step 2] Normalizing pixel values [0,255] -> [0,1]...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(f"After normalization: min={x_train.min():.1f}, max={x_train.max():.1f}")

# One-hot encode labels
print("\n[Step 3] One-hot encoding labels...")
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
print(f"Label 5 -> {y_train_cat[0]}")

# ============================================================================
# PART 3: Build CNN Architecture
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Building CNN (Applying Week 1 Concepts)")
print("=" * 70)

print("\n[Architecture Design]")
print("Conv Block 1: Conv2D(32) -> ReLU -> MaxPool")
print("Conv Block 2: Conv2D(64) -> ReLU -> MaxPool")
print("Flatten: 2D feature maps -> 1D vector (using layers.Flatten!)")
print("Dense: 64 neurons -> 10 output (Softmax)")

model = keras.Sequential([
    # ---- Conv Block 1 ----
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), name='conv1'),
    layers.Activation('relu', name='relu1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    
    # ---- Conv Block 2 ----
    layers.Conv2D(64, (3, 3), padding='same', name='conv2'),
    layers.Activation('relu', name='relu2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    
    # ---- Flatten (proper way, not numpy reshape!) ----
    layers.Flatten(name='flatten'),
    
    # ---- Dense classifier ----
    layers.Dense(64, activation='relu', name='dense1'),
    layers.Dense(10, activation='softmax', name='output')
], name='mnist_cnn')

# Display architecture
print("\n" + "=" * 50)
model.summary()
print("=" * 50)

# Compare parameters to MLP
cnn_params = model.count_params()
mlp_params = 784*128 + 128 + 128*64 + 64 + 64*10 + 10  # Equivalent MLP
print(f"\n[Parameter Comparison]")
print(f"  CNN parameters: {cnn_params:,}")
print(f"  MLP equivalent: {mlp_params:,}")
print(f"  Reduction: {(1 - cnn_params/mlp_params)*100:.1f}%")
print("  -> CNNs are more efficient due to weight sharing!")

# ============================================================================
# PART 4: Compile Model
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Compiling Model")
print("=" * 70)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nConfiguration:")
print("  Optimizer: Adam")
print("  Loss: Categorical Cross-Entropy")
print("  Metrics: Accuracy")
print("\n[OK] Model compiled!")

# ============================================================================
# PART 5: Train with TensorBoard + Early Stopping
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Training the CNN")
print("=" * 70)

# Set up callbacks
log_dir = "logs/mnist_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print(f"\nCallbacks:")
print(f"  TensorBoard: {log_dir}")
print(f"  EarlyStopping: patience=3, restore_best_weights=True")
print(f"\nTraining configuration:")
print(f"  Epochs: 20 (with early stopping)")
print(f"  Batch size: 128")
print(f"  Validation split: 20%")

# Train
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tensorboard_callback, early_stopping]
)

# ============================================================================
# PART 6: Training Analysis
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Training Performance")
print("=" * 70)

epochs_trained = len(history.history['loss'])
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nTraining completed in {epochs_trained} epochs")
print(f"  Training Accuracy: {final_train_acc*100:.2f}%")
print(f"  Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"  Gap: {(final_train_acc - final_val_acc)*100:.2f}%")

print(f"\nView training curves in TensorBoard:")
print(f"  tensorboard --logdir=logs/mnist_cnn")

# ============================================================================
# PART 7: Evaluate on Test Set
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Test Set Evaluation")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

print(f"\nTest Performance:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")

# Predictions
predictions = model.predict(x_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# ============================================================================
# PART 8: Confusion Matrix
# ============================================================================

print("\n" + "=" * 70)
print("PART 8: Confusion Matrix")
print("=" * 70)

cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('CNN Confusion Matrix - MNIST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("[OK] Saved to: confusion_matrix.png")

# Classification report
print("\n" + classification_report(y_test, predicted_classes))

# Misclassifications
misclassified = np.where(predicted_classes != y_test)[0]
print(f"Misclassifications: {len(misclassified)} / {len(y_test)} ({len(misclassified)/len(y_test)*100:.2f}%)")

# ============================================================================
# PART 9: Visualize Misclassifications
# ============================================================================

print("\n" + "=" * 70)
print("PART 9: Misclassified Examples")
print("=" * 70)

n_examples = min(10, len(misclassified))
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(n_examples):
    idx = misclassified[i]
    ax = axes[i // 5, i % 5]
    ax.imshow(x_test[idx].squeeze(), cmap='gray')
    ax.set_title(f'True: {y_test[idx]}\nPred: {predicted_classes[idx]}', color='red')
    ax.axis('off')

plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig('misclassifications.png', dpi=150)
print("[OK] Saved to: misclassifications.png")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: MNIST CNN Classifier")
print("=" * 70)

print(f"""
Key Takeaways:
1. CNN achieved {test_accuracy*100:.2f}% test accuracy (better than MLP!)
2. Only {cnn_params:,} parameters (CNNs use weight sharing)
3. Training converged in {epochs_trained} epochs (early stopping worked)
4. layers.Flatten() properly transitions from 2D conv maps to 1D dense

Why CNN > MLP for images:
- Preserves spatial structure (no flattening input!)
- Weight sharing reduces parameters
- Translation invariance via pooling
- Hierarchical feature learning (edges -> shapes -> objects)

Generated Files:
- mnist_samples.png: Sample training images
- confusion_matrix.png: Per-digit accuracy
- misclassifications.png: Error examples
- TensorBoard logs: logs/mnist_cnn/
""")

print("=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)
print("""
Discussion Points:
1. Why add channel dimension? (Conv2D expects 4D input)
2. Why BatchNorm after Conv? (Stabilizes training, allows higher LR)
3. Where does Flatten go? (Between conv and dense layers)
4. Why dropout before output? (Regularization)

Exercises:
- Add a third conv block - does accuracy improve?
- Try kernel sizes 5x5 instead of 3x3
- Remove BatchNorm - what happens to training?
- Compare training time to MLP
""")