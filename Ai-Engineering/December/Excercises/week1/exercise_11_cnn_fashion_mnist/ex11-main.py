"""
Exercise 11: CNN for Fashion-MNIST

HOW TO RUN:
1. Install required dependencies:
   pip install -r requirements.txt
   
   Or install individually:
   pip install numpy matplotlib tensorflow

2. Run the script:
   python3 ex11-main.py

NOTE: Multiple matplotlib windows will open. Close each window to proceed
      to the next visualization, or run in a non-interactive environment.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


# ================================================
# Part 1: Data Loading
# ================================================
# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Class names for visualization
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# TODO: Explore the data
print(f"Training: {X_train.shape}")
print(f"Test: {X_test.shape}")

# TODO: Visualize samples from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(class_names[i])
    ax.axis('off')
plt.tight_layout()
plt.show()

# ================================================
# Part 2: Preprocessing
# ================================================
# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add channel dimension for CNN: (n, 28, 28) -> (n, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Training shape: {X_train.shape}")

# ================================================
# Part 3: Build the CNN
# ================================================
# Architecture: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Flatten -> Dense -> Output
model = keras.Sequential([
    # Block 1: Convolution + Pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2: Convolution + Pooling
    # TODO: Add second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten and classify
    layers.Flatten(),
    # TODO: Add dense layers
    layers.Dense(64, activation='relu'),
    
    # Output layer
    layers.Dense(10, activation='softmax')
], name='fashion_cnn')

model.summary()

# Task 3.2: Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#=================================================
# Part 4: Train
# ================================================
# Train with validation split
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ================================================  
# Part 5: Evaluate
# ================================================
# Task 5.1: Test Performance
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2%}")

# Check if target accuracy achieved
if test_acc >= 0.90:
    print("SUCCESS! Target accuracy achieved!")
else:
    print(f"Keep trying! Need {(0.90 - test_acc):.2%} more accuracy")

# Task 5.2: Per-Class Accuracy
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Task 5.3: Confusion Matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ================================================
# Part 6: Architecture Experiments
# ================================================
print("\n" + "="*60)
print("Part 6: Architecture Experiments")
print("="*60)

# Record baseline results
baseline_params = model.count_params()
baseline_acc = test_acc
print(f"\nBaseline Results:")
print(f"  Test Accuracy: {baseline_acc:.2%}")
print(f"  Parameters: {baseline_params:,}")

# Variation A: More Filters
print("\n" + "-"*60)
print("Variation A: More Filters")
print("-"*60)
model_a = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
], name='variation_a')

model_a.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel A Architecture:")
model_a.summary()

history_a = model_a.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

test_loss_a, test_acc_a = model_a.evaluate(X_test, y_test, verbose=0)
params_a = model_a.count_params()
print(f"\nVariation A Results:")
print(f"  Test Accuracy: {test_acc_a:.2%}")
print(f"  Parameters: {params_a:,}")

# Variation B: Deeper Network
print("\n" + "-"*60)
print("Variation B: Deeper Network")
print("-"*60)
model_b = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
], name='variation_b')

model_b.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel B Architecture:")
model_b.summary()

history_b = model_b.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

test_loss_b, test_acc_b = model_b.evaluate(X_test, y_test, verbose=0)
params_b = model_b.count_params()
print(f"\nVariation B Results:")
print(f"  Test Accuracy: {test_acc_b:.2%}")
print(f"  Parameters: {params_b:,}")

# Record Results Table
print("\n" + "="*60)
print("Results Summary Table")
print("="*60)
print(f"{'Variation':<12} {'Test Accuracy':<15} {'Parameters':<15} {'Notes':<30}")
print("-"*60)
print(f"{'Baseline':<12} {baseline_acc:<15.2%} {baseline_params:<15,} {'Conv2D(32,64), Dense(64)':<30}")
print(f"{'A':<12} {test_acc_a:<15.2%} {params_a:<15,} {'More filters: Conv2D(64,128), Dense(128)':<30}")
print(f"{'B':<12} {test_acc_b:<15.2%} {params_b:<15,} {'Deeper: 2 conv layers per block':<30}")
print(f"{'C':<12} {'N/A':<15} {'N/A':<15} {'Not implemented':<30}")
print("="*60)

#================================================
# Bonus: Compare to Dense Network
#================================================
print("\n" + "-"*60)
print("Bonus: Compare to Dense Network")
print("-"*60)
# Build Dense-only network for comparison
model_dense = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_dense.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

history_dense = model_dense.fit(X_train, y_train, epochs=15, 
                                validation_split=0.1, verbose=0)

dense_acc = model_dense.evaluate(X_test, y_test, verbose=0)[1]
print(f"Dense Network Test Accuracy: {dense_acc:.2%}")
print(f"CNN Test Accuracy: {test_acc:.2%}")
print(f"CNN improvement: {(test_acc - dense_acc):.2%}")