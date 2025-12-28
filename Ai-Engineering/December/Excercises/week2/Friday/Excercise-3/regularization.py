"""
Exercise 03: Combat Overfitting - Starter Code

Apply multiple regularization techniques to an overfitting model.

Prerequisites:
- Reading: 03-early-stopping-callbacks.md
- Reading: 04-overfitting-mitigation-strategies.md
- Demo: demo_04_regularization_techniques.py (KEY REFERENCE)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# SETUP: Overfitting Scenario (PROVIDED)
# ============================================================================

def create_overfitting_data():
    """Create small dataset that's easy to overfit."""
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    
    X = np.random.randn(n_samples, n_features).astype('float32')
    # Only first 5 features matter
    y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.5*X[:, 3] + 
         np.random.randn(n_samples)*0.5 > 0).astype('float32')
    
    return (X[:400], y[:400]), (X[400:], y[400:])


def create_model(regularization=None, dropout_rate=0.0):
    """
    Create model with optional regularization.
    
    Args:
        regularization: L1/L2 regularizer or None
        dropout_rate: Dropout probability (0 = no dropout)
    """
    model = keras.Sequential()
    
    model.add(layers.Dense(256, activation='relu', input_shape=(50,),
                          kernel_regularizer=regularization))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(128, activation='relu',
                          kernel_regularizer=regularization))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(64, activation='relu',
                          kernel_regularizer=regularization))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# TASK 3.1: Demonstrate Overfitting
# ============================================================================

def demonstrate_overfitting():
    """
    Train baseline without regularization to show overfitting.
    
    OVERFITTING SYMPTOMS:
    - Train accuracy keeps increasing
    - Val accuracy plateaus or decreases
    - Large gap between train and val accuracy
    
    Train for 100 epochs to clearly see the problem.
    """
    print("Task 3.1: Baseline (Overfitting)")
    
    # YOUR CODE:
    # 1. Create model with no regularization
    # 2. Train for 100 epochs
    # 3. Print final train/val accuracy and the gap
    pass


# ============================================================================
# TASK 3.2: Apply Dropout
# ============================================================================

def apply_dropout():
    """
    Add dropout regularization.
    
    DROPOUT CONCEPT:
    - Randomly set fraction of neurons to 0 during training
    - Forces network to be redundant (not rely on single features)
    - Common rates: 0.2-0.5
    
    TEST: dropout_rate=0.3
    """
    print("Task 3.2: Dropout")
    
    # YOUR CODE:
    # model = create_model(dropout_rate=0.3)
    # Train and compare to baseline
    pass


# ============================================================================
# TASK 3.3: Apply L2 Regularization
# ============================================================================

def apply_l2():
    """
    Add L2 weight regularization.
    
    L2 CONCEPT:
    - Add penalty: loss += lambda * sum(weights^2)
    - Encourages smaller weights (smoother function)
    - Prevents extreme weight values
    
    USAGE:
    regularizers.l2(0.01)  # lambda = 0.01
    
    TEST: l2_lambda=0.01
    """
    print("Task 3.3: L2 Regularization")
    
    # YOUR CODE:
    # model = create_model(regularization=regularizers.l2(0.01))
    # Train and compare
    pass


# ============================================================================
# TASK 3.4: Apply Early Stopping
# ============================================================================

def apply_early_stopping():
    """
    Use early stopping to prevent overfitting.
    
    EarlyStopping PARAMETERS:
    - monitor='val_loss': Watch validation loss
    - patience=10: Stop after 10 epochs without improvement
    - restore_best_weights=True: Go back to best epoch
    
    SEE: demo_03_early_stopping.py for usage
    """
    print("Task 3.4: Early Stopping")
    
    # YOUR CODE:
    # early_stop = keras.callbacks.EarlyStopping(...)
    # model.fit(..., callbacks=[early_stop])
    pass


# ============================================================================
# TASK 3.5: Combined Regularization
# ============================================================================

def combined_regularization():
    """
    Apply all techniques together.
    
    COMBINE:
    - Dropout (0.3)
    - L2 regularization (0.01)
    - Early stopping (patience=15)
    
    This is the typical production approach.
    """
    print("Task 3.5: Combined Regularization")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 3.6: Comparison Plot
# ============================================================================

def plot_all_results(histories):
    """
    Plot val_accuracy for all methods on same graph.
    
    INCLUDE:
    - Baseline (overfitting)
    - Dropout only
    - L2 only
    - Early stopping only
    - Combined
    
    Save to: regularization_comparison.png
    """
    plt.figure(figsize=(10, 6))
    
    # YOUR CODE: Plot each history
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Regularization Techniques Comparison')
    plt.legend()
    plt.savefig('regularization_comparison.png')
    print("Saved: regularization_comparison.png")


def write_analysis():
    """
    Write analysis.txt answering:
    
    1. Which technique was most effective alone?
    2. Did combining techniques help?
    3. What was the train/val gap for each?
    4. What's the tradeoff between regularization and model capacity?
    """
    # YOUR CODE
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03: Combat Overfitting")
    print("=" * 60)
    
    # Uncomment as you complete:
    # h_baseline = demonstrate_overfitting()
    # h_dropout = apply_dropout()
    # h_l2 = apply_l2()
    # h_early = apply_early_stopping()
    # h_combined = combined_regularization()
    # plot_all_results({'Baseline': h_baseline, 'Dropout': h_dropout, ...})
    # write_analysis()
