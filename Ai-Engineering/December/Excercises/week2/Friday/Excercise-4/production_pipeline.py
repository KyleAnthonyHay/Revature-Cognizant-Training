"""
Exercise 04: Production Pipeline - Starter Code

Build a complete production-ready training pipeline.

This is a capstone exercise combining all Friday topics.

Prerequisites:
- All Friday readings and demos
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import os
import json
import datetime


# ============================================================================
# PRODUCTION TRAINER CLASS
# ============================================================================

class ProductionTrainer:
    """
    Production-ready training pipeline.
    
    FEATURES TO IMPLEMENT:
    1. Automatic checkpointing (save best model)
    2. Early stopping (prevent overfitting)
    3. TensorBoard logging (visualization)
    4. Model versioning (timestamped runs)
    5. Configuration saving (reproducibility)
    6. Regularization (dropout + L2)
    
    DIRECTORY STRUCTURE:
    production_runs/
      modelname_20240115_143000/
        checkpoints/
          best_model.keras
        logs/
          train/
          validation/
        models/
          final_model.keras
        config.json
        summary.json
    """
    
    def __init__(self, model_name, output_dir='production_runs'):
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Versioned run directory
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(output_dir, f'{model_name}_{self.timestamp}')
        
        # Subdirectories
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.log_dir = os.path.join(self.run_dir, 'logs')
        self.model_dir = os.path.join(self.run_dir, 'models')
        
        self.model = None
        self.history = None
        self.config = {}
    
    def _setup_directories(self):
        """
        Create all output directories.
        Use os.makedirs(path, exist_ok=True)
        """
        # YOUR CODE HERE
        pass
    
    def build_model(self, input_shape, num_classes, 
                    hidden_layers=[128, 64], 
                    dropout_rate=0.3, 
                    l2_lambda=0.001):
        """
        Build model with regularization.
        
        ARCHITECTURE:
        Input -> [Dense + Dropout] x N -> Output
        
        STORE CONFIG for reproducibility:
        self.config = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'l2_lambda': l2_lambda
        }
        """
        # YOUR CODE HERE
        pass
    
    def _create_callbacks(self, patience=10):
        """
        Create production callbacks.
        
        CALLBACKS TO CREATE:
        1. ModelCheckpoint - save best model to checkpoint_dir
        2. EarlyStopping - patience epochs, restore_best_weights=True
        3. TensorBoard - log to log_dir
        4. CSVLogger - save metrics to CSV
        5. ReduceLROnPlateau - reduce LR when stuck
        
        SEE: demo_02_checkpoint_callback.py for ModelCheckpoint
        SEE: demo_03_early_stopping.py for EarlyStopping
        """
        callbacks = []
        
        # YOUR CODE: Create and append each callback
        
        return callbacks
    
    def train(self, x_train, y_train, x_val, y_val,
              epochs=100, batch_size=32, patience=10):
        """
        Run training with all callbacks.
        
        STEPS:
        1. Setup directories
        2. Create callbacks
        3. Train model
        4. Save final model
        5. Save config and summary
        """
        # YOUR CODE HERE
        pass
    
    def _save_config(self):
        """
        Save config.json with all hyperparameters.
        Use json.dump(self.config, f, indent=2)
        """
        # YOUR CODE HERE
        pass
    
    def _save_summary(self):
        """
        Save summary.json with training results.
        
        INCLUDE:
        - Best val_accuracy
        - Best val_loss
        - Total epochs trained
        - Early stopped epoch (if applicable)
        """
        # YOUR CODE HERE
        pass
    
    def load_best_model(self):
        """
        Load the best checkpoint.
        Return: loaded model
        """
        # YOUR CODE HERE
        pass


# ============================================================================
# TEST THE PIPELINE
# ============================================================================

def test_production_pipeline():
    """
    Test the complete pipeline with MNIST.
    
    STEPS:
    1. Load MNIST data
    2. Create ProductionTrainer
    3. Build model
    4. Train
    5. Load best model and evaluate on test set
    6. Print summary of created files
    """
    print("=" * 60)
    print("Testing Production Pipeline")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Use subset for faster testing
    x_train, y_train = x_train[:5000], y_train[:5000]
    
    # Validation split
    x_val, y_val = x_train[-1000:], y_train[-1000:]
    x_train, y_train = x_train[:-1000], y_train[:-1000]
    
    # YOUR CODE:
    # 1. Create trainer
    # trainer = ProductionTrainer('mnist_classifier')
    
    # 2. Build model
    # trainer.build_model(input_shape=(784,), num_classes=10)
    
    # 3. Train
    # trainer.train(x_train, y_train, x_val, y_val, epochs=50)
    
    # 4. Load best and evaluate
    # best_model = trainer.load_best_model()
    # test_loss, test_acc = best_model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {test_acc:.4f}")
    
    # 5. List created files
    # print("\\nCreated files:")
    # for root, dirs, files in os.walk(trainer.run_dir):
    #     for f in files:
    #         print(f"  {os.path.join(root, f)}")
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 04: Production Pipeline")
    print("=" * 60)
    
    # Uncomment to test:
    # test_production_pipeline()
