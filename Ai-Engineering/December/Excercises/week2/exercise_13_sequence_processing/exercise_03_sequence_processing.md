# Exercise 03: Sequence Processing Pipeline

## Learning Objectives

- Handle variable-length sequences with padding and truncation
- Implement masking for padded sequences
- Build complete preprocessing pipelines for RNN/LSTM models
- Understand the impact of preprocessing choices on model performance

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_04_sequence_padding.py`, you saw how to handle variable-length sequences. This exercise has you build a production-quality preprocessing pipeline.

---

## Part 1: Padding Strategies (15 min)

### Task 1.1: Compare Pre vs Post Padding

For classification tasks, pre-padding is typically preferred. Understand why:

```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Variable length sequences
sequences = [
    [1, 2, 3],
    [4, 5, 6, 7, 8],
    [9, 10]
]

# Pre-padding (default): pad at beginning
pre_padded = pad_sequences(sequences, maxlen=6, padding='pre')
print("Pre-padded:")
print(pre_padded)

# Post-padding: pad at end
post_padded = pad_sequences(sequences, maxlen=6, padding='post')
print("\nPost-padded:")
print(post_padded)
```

Implement a function to visualize why pre-padding works better for classification:

```python
def explain_padding_difference():
    """
    Explain why pre-padding is better for classification.
    
    With pre-padding:
    - Actual content is at the END of the sequence
    - RNN's final hidden state captures actual content
    - Final state is used for classification
    
    With post-padding:
    - Actual content is at the BEGINNING
    - RNN processes padding AFTER content
    - Final state is influenced by padding zeros
    """
    # Demonstrate with a simple example
    pass
```

### Task 1.2: Truncation Strategies

When sequences exceed max_length, you must truncate:

```python
long_sequences = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
]

# Pre-truncation: keep end of sequence
pre_truncated = pad_sequences(long_sequences, maxlen=5, truncating='pre')
print("Pre-truncated (keep end):")
print(pre_truncated)

# Post-truncation: keep beginning of sequence
post_truncated = pad_sequences(long_sequences, maxlen=5, truncating='post')
print("\nPost-truncated (keep beginning):")
print(post_truncated)
```

When would you use each strategy?

---

## Part 2: Masking Implementation (15 min)

### Task 2.1: Understanding Masking

Masking tells the RNN to ignore padded positions:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Without masking: RNN processes 0s as regular input
# With masking: RNN skips 0s entirely

def build_masked_model(vocab_size, max_length):
    """Build model with masking enabled."""
    model = keras.Sequential([
        layers.Embedding(
            vocab_size, 
            64, 
            input_length=max_length,
            mask_zero=True  # Enable masking
        ),
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_unmasked_model(vocab_size, max_length):
    """Build model without masking."""
    model = keras.Sequential([
        layers.Embedding(vocab_size, 64, input_length=max_length),
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

### Task 2.2: Compare Masked vs Unmasked

Train both models and compare performance:

```python
# Create synthetic data with varying lengths
def create_synthetic_data(n_samples, max_length, vocab_size):
    """Create data where short and long sequences have different labels."""
    X = []
    y = []
    
    for _ in range(n_samples):
        # Short positive sequences
        if np.random.random() > 0.5:
            length = np.random.randint(5, 15)
            seq = np.random.randint(1, vocab_size, length)
            label = 1
        # Long negative sequences
        else:
            length = np.random.randint(40, 50)
            seq = np.random.randint(1, vocab_size, length)
            label = 0
        
        X.append(seq.tolist())
        y.append(label)
    
    return X, np.array(y)

# Create and pad data
X, y = create_synthetic_data(2000, 50, 1000)
X_padded = pad_sequences(X, maxlen=50, padding='pre')

# Split
split = 1500
X_train, X_test = X_padded[:split], X_padded[split:]
y_train, y_test = y[:split], y[split:]

# Train and compare both models
# Record accuracy difference
```

---

## Part 3: Complete Pipeline Class (15 min)

### Task 3.1: Build Production Pipeline

Create a reusable preprocessing class:

```python
class SequencePreprocessor:
    """
    Complete preprocessing pipeline for sequence data.
    
    Handles:
    - Tokenization
    - Vocabulary building with size limits
    - Padding and truncation
    - Train/test consistency
    """
    
    def __init__(self, max_vocab_size=10000, max_length=None, 
                 padding='pre', truncating='pre'):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.padding = padding
        self.truncating = truncating
        self.tokenizer = None
        self.fitted = False
    
    def fit(self, texts):
        """Fit tokenizer on training texts."""
        from tensorflow.keras.preprocessing.text import Tokenizer
        
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token='<UNK>'
        )
        self.tokenizer.fit_on_texts(texts)
        
        # If max_length not set, use 95th percentile
        if self.max_length is None:
            sequences = self.tokenizer.texts_to_sequences(texts)
            lengths = [len(s) for s in sequences]
            self.max_length = int(np.percentile(lengths, 95))
        
        self.fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts to padded sequences."""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding=self.padding,
            truncating=self.truncating
        )
        return padded
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)
    
    @property
    def vocab_size(self):
        """Return vocabulary size."""
        if self.tokenizer is None:
            return 0
        return min(len(self.tokenizer.word_index) + 1, self.max_vocab_size)
    
    def get_config(self):
        """Return configuration for model building."""
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'embedding_dim': 128  # Suggested default
        }
```

### Task 3.2: Test Complete Pipeline

```python
# Sample movie reviews
train_texts = [
    "This movie was absolutely fantastic! Great acting.",
    "Terrible film. Complete waste of time.",
    "An average movie, nothing special.",
    "One of the best I've seen in years!",
    "Boring and predictable storyline.",
    # ... add more
]

train_labels = [1, 0, 0, 1, 0]  # 1=positive, 0=negative

# Test pipeline
preprocessor = SequencePreprocessor(max_vocab_size=1000)
X_train = preprocessor.fit_transform(train_texts)

print(f"Config: {preprocessor.get_config()}")
print(f"Processed shape: {X_train.shape}")

# Build model using config
config = preprocessor.get_config()
model = keras.Sequential([
    layers.Embedding(config['vocab_size'], config['embedding_dim'], 
                     input_length=config['max_length'], mask_zero=True),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## Reflection Questions

1. **Why is pre-padding preferred for classification?** Think about where the final hidden state comes from.

2. **When would post-padding be better?** Consider sequence-to-sequence tasks.

3. **What happens if you don't use masking with heavily padded data?** Does it hurt performance?

4. **How would you handle a dataset where 5% of sequences are 10x longer than average?** Truncate? Separate model?

---

## Definition of Done

- [ ] Compared pre vs post padding effects
- [ ] Implemented and tested masking
- [ ] Built complete SequencePreprocessor class
- [ ] Verified pipeline works end-to-end
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_03_solution.py`. Only check after completing your attempt!

