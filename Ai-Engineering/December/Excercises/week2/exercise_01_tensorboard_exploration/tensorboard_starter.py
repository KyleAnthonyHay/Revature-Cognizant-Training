"""
Exercise 01: TensorBoard Exploration - Starter Code

Prerequisites:
- Reading: 01-tensorboard-visualization.md
- Demo: demo_01_tensorboard_setup.py (REFERENCE FOR ALL TASKS)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime
import os

# ============================================================================
# DATA & MODEL (PROVIDED - DO NOT MODIFY)
# ============================================================================

def load_mnist_data():
    """Load and preprocess MNIST data"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Create a simple MLP for MNIST classification"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# TASK 1.1: Basic TensorBoard Setup
# ============================================================================

def create_tensorboard_callback(experiment_name="default"):
    """
    Create a TensorBoard callback with proper log directory.
    
    REQUIREMENTS:
    - Log directory format: logs/{experiment_name}_{timestamp}
    - Enable histogram_freq=1 for weight histograms
    - Enable write_graph=True for model visualization
    
    HINTS:
    - Use datetime.datetime.now().strftime("%Y%m%d-%H%M%S") for timestamp
    - keras.callbacks.TensorBoard takes log_dir as first argument
    
    SEE: demo_01_tensorboard_setup.py lines 35-50
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{experiment_name}_{timestamp}"
    return keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )


# ============================================================================
# TASK 1.2: Custom Metric Logging
# ============================================================================

class CustomMetricsCallback(keras.callbacks.Callback):
    """
    Custom callback to log additional metrics to TensorBoard.
    
    REQUIREMENTS:
    - Log current learning rate each epoch
    - Log loss ratio (train_loss / val_loss) as overfitting indicator
    
    HINTS:
    - Create file writer with: tf.summary.create_file_writer(log_dir + "/custom")
    - Get LR: self.model.optimizer.learning_rate (may need .numpy())
    - Write scalar: tf.summary.scalar('name', value, step=epoch)
    - Use "with self.file_writer.as_default():" context
    
    SEE: demo_01_tensorboard_setup.py lines 70-100 for custom callback pattern
    """
    
    def __init__(self, log_dir):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/custom")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log custom metrics at end of each epoch"""
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        loss_ratio = train_loss / val_loss if val_loss else 0.0
        
        with self.file_writer.as_default():
            tf.summary.scalar('learning_rate', lr, step=epoch)
            tf.summary.scalar('loss_ratio', loss_ratio, step=epoch)


# ============================================================================
# TASK 1.3: Experiment Comparison
# ============================================================================

def run_experiment(learning_rate, experiment_name):
    """
    Run training experiment with specified learning rate.
    
    HINTS:
    - Create optimizer: keras.optimizers.Adam(learning_rate=learning_rate)
    - Train for 20 epochs with validation_split=0.2
    
    SEE: demo_01_tensorboard_setup.py "Part 4: Running Multiple Experiments"
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    tb_callback = create_tensorboard_callback(experiment_name)
    (x_train, y_train), _ = load_mnist_data()
    
    model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[tb_callback], verbose=0)


def compare_learning_rates():
    """
    Compare learning rates: 0.001, 0.01, 0.1
    
    After running, view in TensorBoard:
      tensorboard --logdir=logs
    
    Use TensorBoard's "Runs" panel to compare experiments side-by-side.
    """
    learning_rates = [0.001, 0.01, 0.1]
    
    for lr in learning_rates:
        experiment_name = f"lr_{lr}"
        print(f"Running experiment with learning rate: {lr}")
        run_experiment(lr, experiment_name)
        print(f"Completed experiment: {experiment_name}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01: TensorBoard Exploration")
    print("=" * 60)
    
    # Uncomment as you complete:
    # Task 1.1
    tb_callback = create_tensorboard_callback("basic_test")
    model = create_model()
    (x_train, y_train), _ = load_mnist_data()
    model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[tb_callback])
    
    # Task 1.2 - Add custom callback
    tb_callback2 = create_tensorboard_callback("custom_test")
    custom_cb = CustomMetricsCallback("logs/custom_test")
    model2 = create_model()
    model2.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[tb_callback2, custom_cb])
    
    # Task 1.3
    compare_learning_rates()
    
    print("\nTo view results: tensorboard --logdir=logs")
    print("Then open http://localhost:6006")
