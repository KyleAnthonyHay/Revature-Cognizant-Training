# Exercise 01: Tokenization Pipeline

## Learning Objectives

- Implement multiple tokenization strategies
- Handle edge cases in text preprocessing
- Build a reusable tokenization pipeline class
- Compare tokenizer outputs on real text

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_01_tokenization_methods.py`, you saw different tokenization approaches. This exercise has you build a complete tokenization pipeline with vocabulary management.

## The Concepts You Need

### Tokenization Levels

```
# Word-level tokenization:
"I love NLP!" -> ["I", "love", "NLP", "!"]

# Character-level tokenization:
"Hello" -> ["H", "e", "l", "l", "o"]

# Subword tokenization (BPE-style):
"unhappiness" -> ["un", "happiness"] or ["un", "happy", "ness"]
```

### Vocabulary Mapping

```
# Create vocabulary from tokens
vocabulary = {"<PAD>": 0, "<UNK>": 1, "the": 2, "cat": 3, ...}

# Convert tokens to indices
tokens = ["the", "cat"]
indices = [vocabulary.get(t, 1) for t in tokens]  # [2, 3]
```

---

## Part 1: Basic Tokenization (15 min)

### Task 1.1: Implement Simple Tokenizer

Navigate to `starter_code/tokenizer_starter.py` and implement basic tokenization:

```python
import re
from collections import Counter

def simple_tokenize(text):
    """
    Basic word tokenization using whitespace and punctuation.
    
    Steps:
    1. Convert to lowercase
    2. Split on whitespace and punctuation
    3. Remove empty tokens
    
    Example:
        "Hello, World!" -> ["hello", "world"]
    
    Args:
        text: Input string
    
    Returns:
        List of tokens
    """
    # TODO: Convert to lowercase
    # text = text.lower()
    
    # TODO: Split on non-alphanumeric characters
    # Hint: Use re.split(r'\W+', text)
    
    # TODO: Remove empty strings from result
    # Hint: Use list comprehension with filter
    
    pass

def tokenize_with_punctuation(text):
    """
    Tokenize while preserving punctuation as separate tokens.
    
    Example:
        "Hello, World!" -> ["hello", ",", "world", "!"]
    
    Hint: Use re.findall(r"\w+|[^\w\s]", text.lower())
    """
    # TODO: Implement
    pass
```

### Task 1.2: Test Your Tokenizer

Test with these edge cases:

```python
test_cases = [
    "Hello, World!",                    # Basic punctuation
    "I'm learning NLP.",                # Contractions
    "Email: test@example.com",          # Special characters
    "The price is $19.99!",             # Numbers and symbols
    "   Multiple   spaces   here   ",   # Extra whitespace
    "UPPERCASE and lowercase",          # Case handling
]

for text in test_cases:
    print(f"Input: {text}")
    print(f"Simple: {simple_tokenize(text)}")
    print(f"With punct: {tokenize_with_punctuation(text)}")
    print()
```

---

## Part 2: Vocabulary Building (15 min)

### Task 2.1: Build Vocabulary from Corpus

```python
class Vocabulary:
    """
    Vocabulary manager for tokenized text.
    
    Special tokens:
    - <PAD>: Padding token (index 0)
    - <UNK>: Unknown token (index 1)
    - <BOS>: Beginning of sequence (index 2)
    - <EOS>: End of sequence (index 3)
    """
    
    def __init__(self, min_freq=1):
        """
        Initialize vocabulary with special tokens.
        
        Args:
            min_freq: Minimum frequency for a token to be included
        """
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.word_freq = Counter()
    
    def build_vocab(self, texts, tokenizer_fn):
        """
        Build vocabulary from a list of texts.
        
        Steps:
        1. Tokenize each text
        2. Count word frequencies
        3. Add words meeting min_freq threshold
        
        Args:
            texts: List of strings
            tokenizer_fn: Function to tokenize each text
        """
        # TODO: Count all words
        # for text in texts:
        #     tokens = tokenizer_fn(text)
        #     self.word_freq.update(tokens)
        
        # TODO: Add words that meet frequency threshold
        # for word, freq in self.word_freq.items():
        #     if freq >= self.min_freq and word not in self.word2idx:
        #         idx = len(self.word2idx)
        #         self.word2idx[word] = idx
        #         self.idx2word[idx] = word
        
        pass
    
    def encode(self, tokens):
        """
        Convert list of tokens to list of indices.
        
        Unknown tokens map to <UNK> (index 1).
        """
        # TODO: Implement
        # return [self.word2idx.get(token, 1) for token in tokens]
        pass
    
    def decode(self, indices):
        """Convert list of indices back to tokens."""
        # TODO: Implement
        # return [self.idx2word.get(idx, "<UNK>") for idx in indices]
        pass
    
    def __len__(self):
        return len(self.word2idx)
```

### Task 2.2: Test Vocabulary

```python
# Sample corpus
corpus = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "A cat and a dog are friends.",
    "The mat is on the floor."
]

vocab = Vocabulary(min_freq=2)
vocab.build_vocab(corpus, simple_tokenize)

print(f"Vocabulary size: {len(vocab)}")
print(f"Words: {list(vocab.word2idx.keys())}")

# Test encoding/decoding
test_text = "The cat is happy"
tokens = simple_tokenize(test_text)
encoded = vocab.encode(tokens)
decoded = vocab.decode(encoded)

print(f"Tokens: {tokens}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

---

## Part 3: Complete Tokenization Pipeline (15 min)

### Task 3.1: Build TextPreprocessor Class

Combine everything into a reusable class:

```python
class TextPreprocessor:
    """
    Complete text preprocessing pipeline.
    
    Usage:
        preprocessor = TextPreprocessor()
        preprocessor.fit(train_texts)
        train_sequences = preprocessor.transform(train_texts)
        test_sequences = preprocessor.transform(test_texts)
    """
    
    def __init__(self, max_vocab_size=10000, min_freq=2, max_seq_length=100):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_seq_length = max_seq_length
        self.vocab = Vocabulary(min_freq=min_freq)
    
    def fit(self, texts):
        """Build vocabulary from training texts."""
        # TODO: Build vocabulary
        # Consider limiting to max_vocab_size most frequent words
        pass
    
    def transform(self, texts):
        """
        Transform texts to padded sequences.
        
        Steps:
        1. Tokenize each text
        2. Encode tokens to indices
        3. Pad/truncate to max_seq_length
        """
        # TODO: Implement
        pass
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
```

### Task 3.2: Test on Real Dataset

```python
# Test with movie review snippets
reviews = [
    "This movie was absolutely fantastic! Great acting and plot.",
    "Terrible film. Waste of time and money.",
    "An average movie. Nothing special but watchable.",
    "One of the best movies I've ever seen!",
    "Boring and predictable. Skip this one."
]

preprocessor = TextPreprocessor(max_seq_length=20)
sequences = preprocessor.fit_transform(reviews)

print(f"Vocabulary size: {len(preprocessor.vocab)}")
print(f"Sequence shape: {sequences.shape}")
print(f"\nSample sequence: {sequences[0]}")
print(f"Decoded: {preprocessor.vocab.decode(sequences[0])}")
```

---

## Reflection Questions

1. **How does min_freq affect vocabulary size?** Test with min_freq=1, 2, 5 and compare.

2. **What happens when a test text contains words not in the vocabulary?** How does <UNK> handling affect model performance?

3. **How do you choose max_seq_length?** What's the trade-off between too short and too long?

4. **Why do we fit on training data only, then transform both train and test?** What would happen if we fit on test data too?

---

## Definition of Done

- [ ] Basic tokenizer implemented and tested
- [ ] Punctuation-preserving tokenizer implemented
- [ ] Vocabulary class with encode/decode working
- [ ] TextPreprocessor class complete
- [ ] Edge cases handled (unknown words, varying lengths)
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_01_solution.py`. Only check after completing your attempt!

