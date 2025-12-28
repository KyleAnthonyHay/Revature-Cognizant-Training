"""
Exercise 01: RNN Text Generation - Starter Code

Build a character-level RNN for text generation.

Prerequisites:
- Reading: 01-recurrent-neural-networks.md
- Demo: demo_01_simple_rnn.py (REFERENCE FOR RNN PATTERNS)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ============================================================================
# SAMPLE TEXT (PROVIDED)
# ============================================================================

SAMPLE_TEXT = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them To die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to Tis a consummation
Devoutly to be wished To die to sleep
To sleep perchance to dream ay theres the rub
For in that sleep of death what dreams may come
"""


# ============================================================================
# TASK 1.1: Character-Level Preprocessing
# ============================================================================

class CharacterTokenizer:
    """
    Character-level tokenizer.
    
    Unlike word tokenizers, this maps individual characters to indices.
    Vocab is small (26 letters + space + punctuation = ~40 chars).
    """
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, text):
        """
        Build character vocabulary.
        
        STEPS:
        1. Get unique characters from text.lower()
        2. Sort them for reproducibility
        3. Create char_to_idx: {char: index}
        4. Create idx_to_char: {index: char}
        """
        # YOUR CODE HERE
        pass
    
    def encode(self, text):
        """Convert text to list of indices"""
        # YOUR CODE: [self.char_to_idx.get(c, 0) for c in text.lower()]
        return []
    
    def decode(self, indices):
        """Convert indices back to text"""
        # YOUR CODE: ''.join([self.idx_to_char.get(i, '?') for i in indices])
        return ''


def create_sequences(text, tokenizer, seq_length=40):
    """
    Create training sequences for next-character prediction.
    
    CONCEPT: Slide a window over the text
    - Input: characters [i : i+seq_length]
    - Target: character at position i+seq_length
    
    EXAMPLE with seq_length=5:
    Text: "hello world"
    Sequence 1: Input="hello", Target=" "
    Sequence 2: Input="ello ", Target="w"
    ...
    
    Returns:
        X: array of shape (num_sequences, seq_length)
        y: array of shape (num_sequences,)
    """
    # YOUR CODE HERE
    return np.array([]), np.array([])


# ============================================================================
# TASK 1.2: Build RNN Generator
# ============================================================================

def build_text_generator(vocab_size, embedding_dim=64, rnn_units=128):
    """
    Build character-level text generation model.
    
    ARCHITECTURE:
    1. Embedding(vocab_size, embedding_dim) - learn char representations
    2. LSTM(rnn_units, return_sequences=False) - process sequence
    3. Dense(vocab_size, softmax) - predict next character
    
    WHY LSTM OVER SIMPLERNN:
    - Text generation needs long-term memory (sentence structure)
    - LSTM handles this better than SimpleRNN
    
    SEE: demo_01_simple_rnn.py for RNN layer usage
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 1.3: Training
# ============================================================================

def train_generator():
    """
    Train the text generator.
    
    STEPS:
    1. Create tokenizer, fit on SAMPLE_TEXT
    2. Create sequences with seq_length=40
    3. Build model
    4. Train for 50 epochs
    
    EXPECTED: Loss should decrease from ~3.5 to ~1.5
    """
    print("Training Text Generator")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 1.4: Text Generation
# ============================================================================

def generate_text(model, tokenizer, seed_text, length=100, temperature=1.0):
    """
    Generate text starting from seed_text.
    
    ALGORITHM (autoregressive generation):
    1. Encode seed_text to indices
    2. For each new character:
       a. Take last seq_length characters as input
       b. Predict probability distribution over next char
       c. Sample from distribution (using temperature)
       d. Append sampled char to sequence
    3. Decode and return generated text
    
    TEMPERATURE:
    - 1.0 = sample from model's distribution
    - <1.0 = more confident (less random)
    - >1.0 = more creative (more random)
    
    SAMPLING WITH TEMPERATURE:
    logits = model.predict(input)  # raw scores
    logits = logits / temperature
    probs = softmax(logits)
    next_char = np.random.choice(vocab_size, p=probs)
    """
    # YOUR CODE HERE
    return ""


def experiment_temperature():
    """
    Generate text at different temperatures: 0.5, 1.0, 1.5
    
    Observe:
    - Low temp: Repetitive but coherent
    - High temp: Creative but may be nonsense
    """
    temperatures = [0.5, 1.0, 1.5]
    seed = "to be or not"
    
    # YOUR CODE: Generate with each temperature, print results
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01: RNN Text Generation")
    print("=" * 60)
    
    # Uncomment as you complete:
    # train_generator()
    # experiment_temperature()
