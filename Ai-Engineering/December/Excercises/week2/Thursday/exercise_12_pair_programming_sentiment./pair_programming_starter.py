"""
PAIR PROGRAMMING: RNN/LSTM Sentiment Classifier

Partners:
- Partner A: _________________________
- Partner B: _________________________
Date: _________________________

ROTATION SCHEDULE:
- Checkpoint 1 (25 min): Partner A drives, B navigates
- Checkpoint 2 (25 min): Partner B drives, A navigates
- Checkpoint 3 (25 min): Partner A drives
- Checkpoint 4 (25 min): Partner B drives
- Analysis: Together

Prerequisites:
- All Thursday readings and demos
- demo_02_lstm_comparison.py (KEY REFERENCE)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# CHECKPOINT 1: Data Pipeline (25 min)
# DRIVER: Partner A
# ============================================================================

def load_imdb_data(vocab_size=10000, max_length=200):
    """
    Load and prepare IMDB sentiment data.
    
    IMDB DATASET INFO:
    - 25,000 training reviews, 25,000 test reviews
    - Already tokenized to integer sequences
    - Labels: 0 = negative, 1 = positive
    
    STEPS:
    1. Load with keras.datasets.imdb.load_data(num_words=vocab_size)
    2. Pad sequences: pad_sequences(x, maxlen=max_length, padding='pre')
    3. Create validation split (last 5000 from training)
    
    SEE: demo_02_lstm_comparison.py for data loading example
    """
    print("CHECKPOINT 1: Data Pipeline")
    print("Driver: Partner A | Navigator: Partner B")
    print("-" * 50)
    
    # YOUR CODE HERE
    # Return: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    pass


# ============================================================================
# CHECKPOINT 2: SimpleRNN Model (25 min)
# DRIVER: Partner B
# ============================================================================

def build_simple_rnn(vocab_size=10000, embedding_dim=64, rnn_units=32):
    """
    Build SimpleRNN for sentiment classification.
    
    ARCHITECTURE:
    Embedding(vocab_size, embedding_dim) 
    -> SimpleRNN(rnn_units) 
    -> Dense(1, sigmoid)
    
    COMPILE WITH:
    - optimizer: 'adam'
    - loss: 'binary_crossentropy'
    - metrics: ['accuracy']
    
    HINT: mask_zero=True in Embedding layer handles padding
    """
    print("\nCHECKPOINT 2: SimpleRNN Model")
    print("Driver: Partner B | Navigator: Partner A")
    print("-" * 50)
    
    # YOUR CODE HERE
    pass


# ============================================================================
# CHECKPOINT 3: LSTM Model (25 min)
# DRIVER: Partner A
# ============================================================================

def build_lstm(vocab_size=10000, embedding_dim=64, lstm_units=32):
    """
    Build LSTM for sentiment classification.
    
    ARCHITECTURE (same as SimpleRNN but LSTM layer):
    Embedding(vocab_size, embedding_dim) 
    -> LSTM(lstm_units) 
    -> Dense(1, sigmoid)
    
    WHY LSTM FOR SENTIMENT:
    - Reviews can be 200+ words
    - Need to remember context from beginning
    - "The movie was great but the ending was terrible" - need both parts!
    """
    print("\nCHECKPOINT 3: LSTM Model")
    print("Driver: Partner A | Navigator: Partner B")
    print("-" * 50)
    
    # YOUR CODE HERE
    pass


# ============================================================================
# CHECKPOINT 4: Training & Comparison (25 min)
# DRIVER: Partner B
# ============================================================================

def train_and_compare(data):
    """
    Train both models and compare.
    
    TRAINING SETTINGS:
    - epochs: 5 (enough to see difference)
    - batch_size: 128
    - Track training time for each
    
    RECORD:
    - Final val_accuracy for each
    - Training time for each
    - Which converged faster?
    """
    print("\nCHECKPOINT 4: Training & Comparison")
    print("Driver: Partner B | Navigator: Partner A")
    print("-" * 50)
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    
    results = {}
    
    # YOUR CODE:
    # 1. Build SimpleRNN model
    # 2. Train with timing
    # 3. Build LSTM model
    # 4. Train with timing
    # 5. Compare and store results
    
    return results


def plot_comparison(history_rnn, history_lstm):
    """
    Plot training curves side by side.
    
    PLOT:
    - Left subplot: Training accuracy (both models)
    - Right subplot: Validation accuracy (both models)
    
    Save to: rnn_vs_lstm_comparison.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # YOUR CODE: Plot accuracies
    
    plt.tight_layout()
    plt.savefig('rnn_vs_lstm_comparison.png')
    print("Saved: rnn_vs_lstm_comparison.png")


# ============================================================================
# ANALYSIS (Together)
# ============================================================================

def write_analysis(results):
    """
    Write analysis.md together answering:
    
    1. Which model achieved higher accuracy?
    2. Which trained faster? Why?
    3. Look at the first 2 epochs - which converged faster?
    4. For 200-word reviews, why does LSTM outperform SimpleRNN?
    5. When would you choose SimpleRNN over LSTM?
    
    DELIVERABLE: analysis.md with your joint analysis
    """
    template = """# Pair Programming Analysis: RNN vs LSTM

## Partners
- Partner A: [name]
- Partner B: [name]

## Results
| Model     | Val Accuracy | Training Time |
|-----------|--------------|---------------|
| SimpleRNN | ???          | ??? seconds   |
| LSTM      | ???          | ??? seconds   |

## Analysis

### 1. Which model achieved higher accuracy and by how much?
[Your answer]

### 2. Training time comparison - which was faster and why?
[Your answer]

### 3. Early convergence - which learned faster in first 2 epochs?
[Your answer]

### 4. Why does LSTM work better for long reviews?
[Hint: Think about vanishing gradients and long-term dependencies]

### 5. When would SimpleRNN be the better choice?
[Your answer]

## Key Takeaway
[One sentence summary of what you learned]
"""
    
    with open('analysis.md', 'w') as f:
        f.write(template)
    print("Template saved to analysis.md - complete together!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PAIR PROGRAMMING: RNN vs LSTM Sentiment Analysis")
    print("=" * 60)
    print("\nRemember to switch drivers every 25 minutes!")
    print("Both partners should understand all code.\n")
    
    # Uncomment as you complete each checkpoint:
    # data = load_imdb_data()
    # rnn_model = build_simple_rnn()
    # lstm_model = build_lstm()
    # results = train_and_compare(data)
    # write_analysis(results)
