# Exercise 01: RNN Text Generation

## Learning Objectives

- Build a character-level RNN for text generation
- Understand sequence-to-sequence prediction
- Implement temperature-based sampling
- Generate creative text from a trained model

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_01_simple_rnn.py`, you saw how RNNs maintain hidden state across time steps. In this exercise, you'll build a character-level language model that learns to generate text.

## How It Works

```
# Character-level language model predicts next character given previous characters

# Training:
Input:  "hell"
Target: "ello"

The model learns: given "h", predict "e"; given "he", predict "l"; etc.

# Generation:
Seed: "the "
Step 1: "the " -> predict 'c' -> "the c"
Step 2: "the c" -> predict 'a' -> "the ca"
Step 3: "the ca" -> predict 't' -> "the cat"
...continue until desired length
```

---

## Part 1: Data Preparation (15 min)

### Task 1.1: Load and Prepare Text

Navigate to `starter_code/rnn_text_gen_starter.py`:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample text corpus (you can use any text file)
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
"""

def prepare_data(text, seq_length=40):
    """
    Prepare training data for character-level model.
    
    Creates input-output pairs where:
    - Input: sequence of characters
    - Output: next character after sequence
    
    Args:
        text: Raw text string
        seq_length: Length of input sequences
    
    Returns:
        X: Input sequences (encoded)
        y: Target characters (encoded)
        char2idx: Character to index mapping
        idx2char: Index to character mapping
    """
    # TODO: Get unique characters
    # chars = sorted(list(set(text)))
    # char2idx = {c: i for i, c in enumerate(chars)}
    # idx2char = {i: c for c, i in char2idx.items()}
    
    # TODO: Create sequences
    # sequences = []
    # targets = []
    # for i in range(len(text) - seq_length):
    #     seq = text[i:i + seq_length]
    #     target = text[i + seq_length]
    #     sequences.append([char2idx[c] for c in seq])
    #     targets.append(char2idx[target])
    
    # TODO: Convert to numpy arrays
    # X = np.array(sequences)
    # y = np.array(targets)
    
    # return X, y, char2idx, idx2char
    pass
```

### Task 1.2: Verify Data Shape

```python
X, y, char2idx, idx2char = prepare_data(text, seq_length=40)

print(f"Vocabulary size: {len(char2idx)}")
print(f"Total sequences: {len(X)}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Show a sample
sample_idx = 0
print(f"\nSample input: {''.join([idx2char[i] for i in X[sample_idx]])}")
print(f"Sample target: {idx2char[y[sample_idx]]}")
```

---

## Part 2: Build the Model (15 min)

### Task 2.1: Create RNN Model

```python
def build_rnn_generator(vocab_size, embedding_dim=64, rnn_units=128, seq_length=40):
    """
    Build character-level RNN for text generation.
    
    Architecture:
    - Embedding layer: Convert character indices to dense vectors
    - SimpleRNN layer: Process sequence with hidden state
    - Dense output: Predict next character probability
    """
    model = keras.Sequential([
        # TODO: Add Embedding layer
        # layers.Embedding(vocab_size, embedding_dim, input_length=seq_length),
        
        # TODO: Add SimpleRNN layer
        # layers.SimpleRNN(rnn_units, return_sequences=False),
        
        # TODO: Add Dense output layer with softmax
        # layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### Task 2.2: Train the Model

```python
vocab_size = len(char2idx)
model = build_rnn_generator(vocab_size, seq_length=40)
model.summary()

# Train
history = model.fit(
    X, y,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Plot training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training Progress')
plt.show()
```

---

## Part 3: Text Generation (20 min)

### Task 3.1: Implement Sampling Function

```python
def sample_with_temperature(predictions, temperature=1.0):
    """
    Sample from prediction distribution with temperature.
    
    Temperature controls randomness:
    - temperature = 1.0: Sample according to model's probabilities
    - temperature < 1.0: More deterministic (picks high-probability characters)
    - temperature > 1.0: More random (flattens probability distribution)
    
    Formula:
    adjusted_probs = softmax(log(probs) / temperature)
    """
    predictions = np.asarray(predictions).astype('float64')
    
    # Apply temperature
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    
    # Sample from distribution
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)
```

### Task 3.2: Implement Generation Function

```python
def generate_text(model, seed_text, char2idx, idx2char, length=200, temperature=1.0):
    """
    Generate text starting from seed.
    
    Steps:
    1. Encode seed text
    2. Predict next character
    3. Append to generated text
    4. Slide window and repeat
    """
    seq_length = model.input_shape[1]
    generated = seed_text
    
    # Pad or truncate seed to seq_length
    if len(seed_text) < seq_length:
        seed_text = ' ' * (seq_length - len(seed_text)) + seed_text
    else:
        seed_text = seed_text[-seq_length:]
    
    for _ in range(length):
        # Encode current sequence
        x = np.array([[char2idx.get(c, 0) for c in seed_text[-seq_length:]]])
        
        # Predict next character
        predictions = model.predict(x, verbose=0)[0]
        next_idx = sample_with_temperature(predictions, temperature)
        next_char = idx2char[next_idx]
        
        # Append and slide window
        generated += next_char
        seed_text = seed_text[1:] + next_char
    
    return generated
```

### Task 3.3: Generate with Different Temperatures

```python
seed = "To be, or not to be"

print("=" * 60)
print("Temperature = 0.5 (Conservative)")
print("=" * 60)
print(generate_text(model, seed, char2idx, idx2char, length=200, temperature=0.5))

print("\n" + "=" * 60)
print("Temperature = 1.0 (Balanced)")
print("=" * 60)
print(generate_text(model, seed, char2idx, idx2char, length=200, temperature=1.0))

print("\n" + "=" * 60)
print("Temperature = 1.5 (Creative)")
print("=" * 60)
print(generate_text(model, seed, char2idx, idx2char, length=200, temperature=1.5))
```

---

## Part 4: Improve with LSTM (10 min)

### Task 4.1: Replace RNN with LSTM

Modify your model to use LSTM instead of SimpleRNN:

```python
def build_lstm_generator(vocab_size, embedding_dim=64, lstm_units=128, seq_length=40):
    """Same architecture but with LSTM instead of SimpleRNN."""
    # TODO: Build model with layers.LSTM instead of layers.SimpleRNN
    pass
```

Train the LSTM model and compare generation quality.

---

## Reflection Questions

1. **How does temperature affect the generated text?** Describe the difference between low and high temperature outputs.

2. **Why does the model sometimes generate repeated patterns?** What causes this and how might you fix it?

3. **How did LSTM generation compare to SimpleRNN?** Did it produce more coherent long-range text?

4. **What would you need to change to train on a much larger corpus (e.g., all of Shakespeare)?**

---

## Definition of Done

- [ ] Data preparation function working
- [ ] RNN model trained successfully
- [ ] Temperature-based sampling implemented
- [ ] Generated text at 3 temperature levels
- [ ] LSTM variant implemented and compared
- [ ] Reflection questions answered

---

## Bonus Challenge

Implement beam search decoding instead of temperature sampling:

```python
def beam_search_generate(model, seed, beam_width=3, length=100):
    """
    Generate text using beam search.
    
    Keep top-k hypotheses at each step instead of sampling.
    """
    # TODO: Implement beam search
    pass
```

---

## Solution

The solution is available in `solutions/exercise_01_solution.py`. Only check after completing your attempt!

