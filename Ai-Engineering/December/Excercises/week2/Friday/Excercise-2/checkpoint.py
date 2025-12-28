"""
Exercise 02: Checkpoint Strategy - Starter Code

Implement optimal checkpointing for long training runs.

Prerequisites:
- Reading: 02-model-checkpoints.md
- Demo: demo_02_checkpoint_callback.py (KEY REFERENCE)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import shutil

# Setup
CKPT_DIR = 'checkpoints'
if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR)


# ============================================================================
# SETUP (PROVIDED)
# ============================================================================

def load_data():
    """Load MNIST subset"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[:10000].reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return (x_train, y_train[:10000]), (x_test, y_test)

def create_model():
    """Create simple model"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# TASK 2.1: Basic Checkpointing
# ============================================================================

def basic_checkpointing():
    """
    Save checkpoint after every epoch.
    
    ModelCheckpoint PARAMETERS:
    - filepath: where to save (use {epoch:02d} for epoch number)
    - save_weights_only: True for smaller files
    - save_freq: 'epoch' or integer (batches)
    - verbose: 1 to print save messages
    
    FILEPATH PATTERN:
    'checkpoints/ckpt_epoch-{epoch:02d}.keras'
    -> Creates: ckpt_epoch-01.keras, ckpt_epoch-02.keras, etc.
    """
    print("Task 2.1: Basic Checkpointing")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 2.2: Save Best Model Only
# ============================================================================

def save_best_only():
    """
    Only save when validation loss improves.
    
    KEY PARAMETERS:
    - monitor='val_loss': Watch validation loss
    - mode='min': Lower is better
    - save_best_only=True: Only save improvements
    
    BENEFIT: Saves disk space, keeps only best model
    """
    print("Task 2.2: Save Best Only")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 2.3: Resume Training from Checkpoint
# ============================================================================

def resume_training():
    """
    Simulate crash recovery: train 5 epochs, "crash", resume.
    
    SCENARIO:
    1. Train for 5 epochs, save checkpoint
    2. Clear model (simulate crash)
    3. Load checkpoint
    4. Continue training for 5 more epochs
    
    KEY: After loading, model should continue improving!
    
    HINT: After keras.models.load_model(), you may need to recompile
    if optimizer state issues occur (Keras 3 quirk).
    """
    print("Task 2.3: Resume Training")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 2.4: Cleanup Old Checkpoints
# ============================================================================

def smart_checkpointing():
    """
    Keep only the N most recent checkpoints to save disk space.
    
    APPROACH:
    1. Use custom callback or post-training cleanup
    2. After each epoch, delete checkpoints older than last N
    
    ALTERNATIVE: Use keras.callbacks.BackupAndRestore for automatic resume
    """
    print("Task 2.4: Smart Checkpointing")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02: Checkpoint Strategy")
    print("=" * 60)
    
    # Uncomment as you complete:
    # basic_checkpointing()
    # save_best_only()
    # resume_training()
    # smart_checkpointing()
