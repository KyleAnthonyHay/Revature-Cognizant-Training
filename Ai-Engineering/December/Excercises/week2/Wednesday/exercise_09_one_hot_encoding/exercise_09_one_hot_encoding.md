# Exercise 02: One-Hot vs. Dense Encoding

## Learning Objectives

- Implement one-hot encoding for text
- Measure memory usage and sparsity
- Demonstrate the semantic similarity problem
- Compare with dense embedding representations

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_02_one_hot_encoding.py`, you saw that one-hot encoding treats all words as equally different. This exercise quantifies the problems and motivates the need for dense embeddings.

## Key Formulas

```
# One-Hot Vector for word i in vocabulary of size V:
# Vector has V dimensions, all zeros except position i = 1
# one_hot("cat") where cat is index 3 in vocab size 5:
# [0, 0, 0, 1, 0]

# Cosine Similarity between vectors A and B:
# cos_sim = (A dot B) / (||A|| * ||B||)
# For one-hot: cos_sim is always 0 between different words (orthogonal vectors)

# Memory for one-hot: V * n_words * sizeof(float)
# Memory for dense: embedding_dim * n_words * sizeof(float)
```

---

## Part 1: Implement One-Hot Encoding (15 min)

### Task 1.1: Create One-Hot Encoder

Navigate to `starter_code/one_hot_starter.py` and implement:

```python
import numpy as np

class OneHotEncoder:
    """
    One-hot encoder for text vocabulary.
    """
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
    
    def fit(self, words):
        """
        Build vocabulary from list of words.
        
        Args:
            words: List of unique words
        """
        # TODO: Create mappings
        # self.word2idx = {word: idx for idx, word in enumerate(words)}
        # self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        # self.vocab_size = len(words)
        pass
    
    def encode_word(self, word):
        """
        Encode single word as one-hot vector.
        
        Returns:
            numpy array of shape (vocab_size,)
        """
        # TODO: Create zero vector and set appropriate index to 1
        # vec = np.zeros(self.vocab_size)
        # if word in self.word2idx:
        #     vec[self.word2idx[word]] = 1
        # return vec
        pass
    
    def encode_sequence(self, words):
        """
        Encode list of words as matrix.
        
        Returns:
            numpy array of shape (len(words), vocab_size)
        """
        # TODO: Stack one-hot vectors
        pass
```

### Task 1.2: Measure Memory Usage

```python
def measure_memory(vocab_size, seq_length, dtype=np.float32):
    """
    Calculate memory usage for one-hot encoding.
    
    Returns:
        Memory in bytes and human-readable format
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = vocab_size * seq_length * bytes_per_element
    
    # TODO: Convert to human-readable format (KB, MB, GB)
    pass

# Test with realistic NLP vocabulary sizes
vocab_sizes = [1000, 10000, 50000, 100000]
seq_length = 100

for v in vocab_sizes:
    mem = measure_memory(v, seq_length)
    print(f"Vocab {v:,}: {mem}")
```

---

## Part 2: The Similarity Problem (15 min)

### Task 2.1: Compute Cosine Similarity

```python
def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    
    Formula: cos_sim = (A dot B) / (||A|| * ||B||)
    
    Returns:
        Similarity score between -1 and 1
    """
    # TODO: Implement cosine similarity
    # Handle division by zero for zero vectors
    pass
```

### Task 2.2: Demonstrate the Problem

```python
# Create encoder with semantic word groups
words = [
    "cat", "dog", "bird",           # Animals
    "car", "truck", "motorcycle",   # Vehicles
    "happy", "sad", "angry",        # Emotions
    "run", "walk", "jump"           # Actions
]

encoder = OneHotEncoder()
encoder.fit(words)

# Compute similarity between all pairs
print("One-Hot Similarities:")
print("-" * 40)

# Compare semantically similar words
pairs_similar = [("cat", "dog"), ("car", "truck"), ("happy", "sad")]
for w1, w2 in pairs_similar:
    v1 = encoder.encode_word(w1)
    v2 = encoder.encode_word(w2)
    sim = cosine_similarity(v1, v2)
    print(f"{w1} vs {w2}: {sim:.4f}")  # Should all be 0!

# Compare semantically different words
pairs_different = [("cat", "car"), ("happy", "truck")]
for w1, w2 in pairs_different:
    v1 = encoder.encode_word(w1)
    v2 = encoder.encode_word(w2)
    sim = cosine_similarity(v1, v2)
    print(f"{w1} vs {w2}: {sim:.4f}")  # Also 0!

print("\nProblem: All similarities are 0!")
print("One-hot cannot distinguish semantic relationships.")
```

---

## Part 3: Compare with Dense Embeddings (15 min)

### Task 3.1: Simulate Dense Embeddings

Create embeddings that capture semantic similarity:

```python
def create_semantic_embeddings(words, embedding_dim=8):
    """
    Create dense embeddings with semantic structure.
    
    Words in similar categories will have similar embeddings.
    """
    np.random.seed(42)
    
    # Define semantic categories
    categories = {
        "animal": ["cat", "dog", "bird"],
        "vehicle": ["car", "truck", "motorcycle"],
        "emotion": ["happy", "sad", "angry"],
        "action": ["run", "walk", "jump"]
    }
    
    # Create category base vectors
    category_vectors = {
        cat: np.random.randn(embedding_dim) for cat in categories
    }
    
    embeddings = {}
    for category, word_list in categories.items():
        base = category_vectors[category]
        for word in word_list:
            # Add small random noise to base vector
            noise = np.random.randn(embedding_dim) * 0.1
            embeddings[word] = base + noise
    
    return embeddings
```

### Task 3.2: Compare Similarities

```python
embeddings = create_semantic_embeddings(words)

print("\nDense Embedding Similarities:")
print("-" * 40)

# Same pairs as before
for w1, w2 in pairs_similar:
    sim = cosine_similarity(embeddings[w1], embeddings[w2])
    print(f"{w1} vs {w2}: {sim:.4f}")  # Should be HIGH (similar categories)

for w1, w2 in pairs_different:
    sim = cosine_similarity(embeddings[w1], embeddings[w2])
    print(f"{w1} vs {w2}: {sim:.4f}")  # Should be LOW (different categories)
```

### Task 3.3: Memory Comparison

```python
def compare_memory(vocab_size, seq_length, embedding_dim=100):
    """Compare memory usage between one-hot and dense embeddings."""
    
    one_hot_bytes = vocab_size * seq_length * 4  # float32
    dense_bytes = embedding_dim * seq_length * 4  # float32
    
    # TODO: Print comparison
    # Include ratio: one_hot / dense
    pass

compare_memory(50000, 100, embedding_dim=100)
# Expected: Dense uses ~500x less memory!
```

---

## Summary Table

Fill in this table with your results:

| Metric | One-Hot | Dense (dim=100) |
|--------|---------|-----------------|
| Memory per word (50K vocab) | | |
| Memory per sequence (100 words) | | |
| Similarity: cat-dog | | |
| Similarity: cat-car | | |
| Captures semantics? | | |

---

## Reflection Questions

1. **What is the sparsity of one-hot vectors?** (What percentage of values are zero?) How does this waste computation?

2. **Why do one-hot vectors have zero similarity?** Think about what the dot product of two one-hot vectors equals.

3. **How do dense embeddings capture "cat is similar to dog"?** What property of the vectors enables this?

4. **If vocabulary size is 100K and embedding dimension is 300, how much more efficient are dense embeddings?**

---

## Definition of Done

- [ ] One-hot encoder implemented and working
- [ ] Memory usage calculated for various vocab sizes
- [ ] Cosine similarity function implemented
- [ ] Demonstrated 0 similarity problem with one-hot
- [ ] Created dense embeddings with semantic structure
- [ ] Memory comparison completed
- [ ] Summary table filled in
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_02_solution.py`. Only check after completing your attempt!

