"""
Exercise 08: Sequential Model Builder
Part 1: Warmup - Model 1: Simple Binary Classifier
Part 2: Architecture Challenges - Models 2-5

HOW TO RUN THIS FILE:
=====================

1. Activate the virtual environment:
   source ../../venv/bin/activate

2. Install dependencies (if not already installed):
   pip install -r requirements.txt

3. Run the script:
   python ex8-main.py
   
   OR
   
   python3 ex8-main.py

4. Deactivate the virtual environment when done:
   deactivate

REQUIREMENTS:
- Python 3.8+
- TensorFlow 2.16.2
- See requirements.txt for full list of dependencies
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("MODEL 1: Simple Binary Classifier")
print("=" * 60)

# Requirements:
# - Input: 20 features
# - Output: Binary classification (0 or 1)
# - At least 2 hidden layers
# - Use ReLU in hidden layers

model_1 = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='binary_classifier')

model_1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_1.summary()

print(f"\nTotal parameters: {model_1.count_params():,}")


# =============================================================================
# MODEL 2: Multi-Class Classifier
# =============================================================================

print("\n" + "=" * 60)
print("MODEL 2: Multi-Class Classifier")
print("=" * 60)

# Requirements:
# - Input: 784 features (flattened 28x28 image)
# - Output: 10 classes
# - Use dropout for regularization
# - Total parameters should be between 100,000 and 200,000

model_2 = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
], name='multiclass_classifier')

model_2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_2.summary()

total_params_2 = model_2.count_params()
print(f"\nTotal parameters: {total_params_2:,}")
assert 100_000 <= total_params_2 <= 200_000, f"Parameters ({total_params_2:,}) out of range!"
print("✓ Parameter count verified!")


# =============================================================================
# MODEL 3: Regression Network
# =============================================================================

print("\n" + "=" * 60)
print("MODEL 3: Regression Network")
print("=" * 60)

# Requirements:
# - Input: 13 features (Boston housing style)
# - Output: Single continuous value (price)
# - Use batch normalization
# - Include at least 3 hidden layers

model_3 = keras.Sequential([
    layers.Dense(64, input_shape=(13,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(16),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(1)
], name='regression_network')

model_3.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nModel Architecture:")
model_3.summary()

print(f"\nTotal parameters: {model_3.count_params():,}")


# =============================================================================
# MODEL 4: Deep Network
# =============================================================================

print("\n" + "=" * 60)
print("MODEL 4: Deep Network")
print("=" * 60)

# Requirements:
# - Input: 100 features
# - Output: 5 classes
# - Must have at least 5 hidden layers
# - Must use both Dropout AND BatchNormalization
# - Parameters between 50,000 and 100,000

model_4 = keras.Sequential([
    layers.Dense(150, input_shape=(100,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(120),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(90),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(60),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(40),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(5, activation='softmax')
], name='deep_network')

model_4.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_4.summary()

total_params_4 = model_4.count_params()
print(f"\nTotal parameters: {total_params_4:,}")
assert 50_000 <= total_params_4 <= 100_000, f"Parameters ({total_params_4:,}) out of range!"
print("✓ Parameter count verified!")


# =============================================================================
# MODEL 5: Minimal Network Challenge
# =============================================================================

print("\n" + "=" * 60)
print("MODEL 5: Minimal Network Challenge")
print("=" * 60)

# Requirements:
# - Input: 50 features
# - Output: 3 classes
# - MAXIMUM 1,000 parameters
# - Must still have at least 1 hidden layer

model_5 = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(50,)),
    layers.Dense(3, activation='softmax')
], name='minimal_network')

model_5.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_5.summary()

total_params_5 = model_5.count_params()
print(f"\nTotal parameters: {total_params_5:,}")
assert total_params_5 <= 1000, f"Too many parameters ({total_params_5:,})!"
print("✓ Parameter count verified!")


# =============================================================================
# PART 3: Reflection and Documentation
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: REFLECTION AND DOCUMENTATION")
print("=" * 60)

# =============================================================================
# Comparison Table
# =============================================================================

print("\n" + "-" * 60)
print("COMPARISON TABLE")
print("-" * 60)

# Get parameter counts
params_1 = model_1.count_params()
params_2 = model_2.count_params()
params_3 = model_3.count_params()
params_4 = model_4.count_params()
params_5 = model_5.count_params()

# Print table header
print(f"\n{'Model':<8} {'Input':<18} {'Output':<20} {'Hidden Layers':<25} {'Parameters':<15} {'Special Features':<30}")
print("-" * 120)

# Model 1
print(f"{'1':<8} {'20 features':<18} {'Binary (0 or 1)':<20} {'2 (64, 32)':<25} {params_1:>13,} {'ReLU activation':<30}")

# Model 2
print(f"{'2':<8} {'784 features':<18} {'10 classes':<20} {'2 (128, 64)':<25} {params_2:>13,} {'Dropout (0.3)':<30}")

# Model 3
print(f"{'3':<8} {'13 features':<18} {'Single value':<20} {'3 (64, 32, 16)':<25} {params_3:>13,} {'BatchNormalization':<30}")

# Model 4
print(f"{'4':<8} {'100 features':<18} {'5 classes':<20} {'5 (150,120,90,60,40)':<25} {params_4:>13,} {'Dropout + BatchNorm':<30}")

# Model 5
print(f"{'5':<8} {'50 features':<18} {'3 classes':<20} {'1 (8)':<25} {params_5:>13,} {'Minimal params (<1000)':<30}")

print("-" * 120)

