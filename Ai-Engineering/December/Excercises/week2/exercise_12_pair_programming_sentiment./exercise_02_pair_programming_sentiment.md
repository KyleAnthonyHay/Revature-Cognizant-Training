# Exercise 02: RNN vs LSTM Sentiment Analysis (PAIR PROGRAMMING)

## Learning Objectives

- Build and compare RNN and LSTM sentiment classifiers
- Practice pair programming with Driver/Navigator roles
- Implement complete NLP pipelines from text to prediction
- Analyze why LSTM outperforms SimpleRNN on longer sequences

## Duration

**Estimated Time:** 90 minutes (3 rotations of 25 minutes + handoff time)

## Pair Programming Rules

### Roles

**Driver:**
- Controls the keyboard
- Writes the code
- Thinks aloud while coding
- Focuses on implementation details

**Navigator:**
- Reviews each line as it's written
- Thinks about the big picture
- Catches bugs and typos
- Suggests improvements
- Keeps track of the overall goal

### Rotation Schedule

| Time | Activity |
|------|----------|
| 0:00-0:05 | Read exercise together, plan approach |
| 0:05-0:30 | Rotation 1: Build preprocessing pipeline |
| 0:30-0:35 | Switch roles, brief handoff |
| 0:35-1:00 | Rotation 2: Build and train models |
| 1:00-1:05 | Switch roles, brief handoff |
| 1:05-1:30 | Rotation 3: Evaluate and compare |

### Handoff Checklist
Before switching:
1. Driver explains current code state
2. Driver shows what's working/not working
3. Navigator asks clarifying questions
4. Both agree on next steps

---

## The Challenge

Build sentiment classifiers for movie reviews using:
1. SimpleRNN architecture
2. LSTM architecture

Compare their performance on the IMDB dataset.

---

## Part 1: Data and Preprocessing (Rotation 1)

### Goal
Build a complete text preprocessing pipeline for IMDB reviews.

### Starter Code

Navigate to `starter_code/pair_programming_starter.py`:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load IMDB dataset
# Note: This is a pre-tokenized version for convenience
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Sample sequence length: {len(x_train[0])}")
```

### Tasks for Rotation 1

**Task 1.1: Analyze Sequence Lengths**

Determine appropriate padding length:

```python
# Calculate sequence length statistics
lengths = [len(seq) for seq in x_train]
print(f"Mean length: {np.mean(lengths):.1f}")
print(f"Median length: {np.median(lengths):.1f}")
print(f"Max length: {max(lengths)}")
print(f"95th percentile: {np.percentile(lengths, 95):.1f}")

# Plot distribution
plt.hist(lengths, bins=50)
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Review Length Distribution')
plt.axvline(np.percentile(lengths, 95), color='r', label='95th percentile')
plt.legend()
plt.show()
```

**Task 1.2: Pad Sequences**

Choose a max_length and pad all sequences:

```python
max_length = ___  # Decide based on analysis above

x_train_padded = pad_sequences(x_train, maxlen=max_length, padding='pre')
x_test_padded = pad_sequences(x_test, maxlen=max_length, padding='pre')

print(f"Padded training shape: {x_train_padded.shape}")
print(f"Padded test shape: {x_test_padded.shape}")
```

**Task 1.3: Create Validation Split**

```python
# Split training data into train and validation
val_size = 5000
x_val = x_train_padded[:val_size]
y_val = y_train[:val_size]
x_train_final = x_train_padded[val_size:]
y_train_final = y_train[val_size:]

print(f"Training: {len(x_train_final)}, Validation: {len(x_val)}, Test: {len(x_test_padded)}")
```

---

## Part 2: Build Models (Rotation 2)

### Goal
Create identical architectures using SimpleRNN and LSTM.

### Tasks for Rotation 2

**Task 2.1: Build SimpleRNN Model**

```python
def build_simple_rnn(vocab_size, embedding_dim, max_length):
    """
    Build SimpleRNN sentiment classifier.
    
    Architecture:
    - Embedding (vocab_size, embedding_dim)
    - SimpleRNN (64 units)
    - Dense (1, sigmoid) for binary classification
    """
    model = keras.Sequential([
        # Your code here
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Task 2.2: Build LSTM Model**

```python
def build_lstm(vocab_size, embedding_dim, max_length):
    """
    Build LSTM sentiment classifier.
    
    Same architecture but with LSTM instead of SimpleRNN.
    """
    model = keras.Sequential([
        # Your code here
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Task 2.3: Train Both Models**

```python
vocab_size = 10000
embedding_dim = 128

# Build models
rnn_model = build_simple_rnn(vocab_size, embedding_dim, max_length)
lstm_model = build_lstm(vocab_size, embedding_dim, max_length)

print("SimpleRNN Model:")
rnn_model.summary()
print("\nLSTM Model:")
lstm_model.summary()

# Train SimpleRNN
print("\nTraining SimpleRNN...")
rnn_history = rnn_model.fit(
    x_train_final, y_train_final,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)

# Train LSTM
print("\nTraining LSTM...")
lstm_history = lstm_model.fit(
    x_train_final, y_train_final,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=1
)
```

---

## Part 3: Evaluate and Compare (Rotation 3)

### Goal
Compare model performance and understand why LSTM performs better.

### Tasks for Rotation 3

**Task 3.1: Plot Training Curves**

```python
def plot_comparison(rnn_history, lstm_history):
    """Plot training curves for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(rnn_history.history['accuracy'], label='RNN Train')
    axes[0].plot(rnn_history.history['val_accuracy'], label='RNN Val')
    axes[0].plot(lstm_history.history['accuracy'], label='LSTM Train')
    axes[0].plot(lstm_history.history['val_accuracy'], label='LSTM Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy Comparison')
    axes[0].legend()
    
    # Loss
    axes[1].plot(rnn_history.history['loss'], label='RNN Train')
    axes[1].plot(rnn_history.history['val_loss'], label='RNN Val')
    axes[1].plot(lstm_history.history['loss'], label='LSTM Train')
    axes[1].plot(lstm_history.history['val_loss'], label='LSTM Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss Comparison')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_comparison(rnn_history, lstm_history)
```

**Task 3.2: Test Set Evaluation**

```python
print("Test Set Evaluation")
print("=" * 40)

rnn_test_loss, rnn_test_acc = rnn_model.evaluate(x_test_padded, y_test, verbose=0)
lstm_test_loss, lstm_test_acc = lstm_model.evaluate(x_test_padded, y_test, verbose=0)

print(f"SimpleRNN - Loss: {rnn_test_loss:.4f}, Accuracy: {rnn_test_acc:.4f}")
print(f"LSTM      - Loss: {lstm_test_loss:.4f}, Accuracy: {lstm_test_acc:.4f}")
print(f"\nImprovement: {(lstm_test_acc - rnn_test_acc) * 100:.2f}%")
```

**Task 3.3: Error Analysis**

```python
# Find examples where models disagree
rnn_preds = (rnn_model.predict(x_test_padded) > 0.5).astype(int).flatten()
lstm_preds = (lstm_model.predict(x_test_padded) > 0.5).astype(int).flatten()

disagreements = np.where(rnn_preds != lstm_preds)[0]
print(f"\nModels disagree on {len(disagreements)} samples ({len(disagreements)/len(y_test)*100:.1f}%)")

# Check which model is correct when they disagree
rnn_correct = np.sum(rnn_preds[disagreements] == y_test[disagreements])
lstm_correct = np.sum(lstm_preds[disagreements] == y_test[disagreements])
print(f"When disagreeing: RNN correct {rnn_correct}, LSTM correct {lstm_correct}")
```

---

## Deliverables

Create a summary document with:

1. **Preprocessing decisions**
   - Chosen max_length and justification
   - Padding strategy (pre vs post)

2. **Model architectures**
   - Parameter counts for each model
   - Training time comparison

3. **Results table**

| Metric | SimpleRNN | LSTM | Difference |
|--------|-----------|------|------------|
| Val Accuracy | | | |
| Test Accuracy | | | |
| Training Time | | | |
| Parameters | | | |

4. **Analysis**
   - Why does LSTM outperform SimpleRNN on this task?
   - At what sequence lengths would SimpleRNN be sufficient?

---

## Reflection Questions (Answer Together)

1. **What did you observe about training curves?** Did either model show signs of overfitting?

2. **Why does LSTM maintain better performance on longer reviews?** Think about gradient flow.

3. **How did pair programming affect your productivity?** What worked well? What was challenging?

4. **If you had more time, what improvements would you make?** (Bidirectional? More layers? Attention?)

---

## Definition of Done

- [ ] Preprocessing pipeline complete with justified max_length
- [ ] Both models trained for 10 epochs
- [ ] Training curves plotted and compared
- [ ] Test accuracy for both models recorded
- [ ] Error analysis completed
- [ ] Summary document written
- [ ] Reflection questions answered (by both partners)

---

## Solution

The solution is available in `solutions/exercise_02_solution.py`. Only check after completing your attempt!

