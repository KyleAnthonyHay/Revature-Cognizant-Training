"""
Exercise 01: Model Persistence - Starter Code

Master different model saving formats.

Prerequisites:
- Reading: 01-saving-loading-models.md
- Demo: demo_01_save_load_models.py (KEY REFERENCE FOR FORMATS)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import shutil

# Setup
SAVE_DIR = 'saved_models'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)


# ============================================================================
# SETUP (PROVIDED)
# ============================================================================

def create_and_train_model():
    """Create and briefly train a model for testing."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[:5000].reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train = y_train[:5000]
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name='mnist_classifier')
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)
    
    return model, (x_test, y_test)


# ============================================================================
# TASK 1.1: Save Complete Models
# ============================================================================

def save_complete_model(model):
    """
    Save model in different formats and compare.
    
    FORMATS TO TEST:
    1. .keras (recommended for Keras 3)
       model.save('model.keras')
       
    2. .h5 (legacy HDF5 format)
       model.save('model.h5')
       
    3. SavedModel directory (for TF Serving)
       model.export('savedmodel/')  # Keras 3
    
    PRINT:
    - File size for each format
    - What's included in each
    
    SEE: demo_01_save_load_models.py for examples
    """
    print("Task 1.1: Saving Complete Models")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 1.2: Save Weights Only
# ============================================================================

def save_weights_only(model):
    """
    Save just the model weights (no architecture).
    
    WHEN TO USE:
    - When you version control your model code
    - For smaller file size
    - For transfer learning
    
    SAVE WITH:
    model.save_weights('model.weights.h5')  # Note: .weights.h5 required in Keras 3
    
    LOAD INTO NEW MODEL:
    new_model = create_model()  # Must match architecture
    new_model.load_weights('model.weights.h5')
    """
    print("Task 1.2: Saving Weights Only")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 1.3: Load and Verify
# ============================================================================

def load_and_verify(model, test_data):
    """
    Load each saved model and verify it matches original.
    
    VERIFICATION:
    1. Evaluate on test data - accuracy should match
    2. Predict on subset - predictions should be identical
    
    LOADING:
    - .keras/.h5: keras.models.load_model(path)
    - weights: model.load_weights(path)
    - SavedModel: keras.layers.TFSMLayer(path, call_endpoint='serve')
    
    TIP: Use np.allclose() to compare predictions
    """
    print("Task 1.3: Load and Verify")
    
    x_test, y_test = test_data
    
    # YOUR CODE:
    # 1. Evaluate original model
    # 2. Load each saved format
    # 3. Evaluate loaded models
    # 4. Verify predictions match
    pass


# ============================================================================
# TASK 1.4: Format Comparison Report
# ============================================================================

def create_comparison_report():
    """
    Create report.md comparing formats.
    
    COMPARE:
    | Format     | Size   | Contains              | Use Case           |
    |------------|--------|----------------------|-------------------|
    | .keras     | ???    | arch + weights + opt | Default choice    |
    | .h5        | ???    | arch + weights + opt | Legacy support    |
    | SavedModel | ???    | TF graph + vars      | TF Serving        |
    | .weights   | ???    | weights only         | Transfer learning |
    """
    # YOUR CODE: Write comparison based on your measurements
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01: Model Persistence")
    print("=" * 60)
    
    # Uncomment as you complete:
    # model, test_data = create_and_train_model()
    # save_complete_model(model)
    # save_weights_only(model)
    # load_and_verify(model, test_data)
    # create_comparison_report()
