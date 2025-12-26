"""
Demo 02: One-Hot Encoding for Text

This demo shows trainees how to:
1. Build one-hot encoders for text data
2. Understand dimensionality and sparsity issues
3. Visualize why one-hot encoding fails to capture semantics
4. Motivate the need for dense embeddings

Learning Objectives:
- Create one-hot vectors from word indices
- Understand limitations: sparsity, no similarity
- Recognize when one-hot encoding is appropriate

References:
- Written Content: 03-one-hot-encoding.md
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: What is One-Hot Encoding?
# ============================================================================

print("=" * 70)
print("PART 1: One-Hot Encoding Basics")
print("=" * 70)

print("\nOne-Hot Encoding: Each word becomes a vector where:")
print("  - Vector length = vocabulary size")
print("  - One position is 1 (the word's index)")
print("  - All other positions are 0")

# Simple vocabulary
vocabulary = ["cat", "dog", "bird", "fish", "horse"]
vocab_size = len(vocabulary)

# Create word-to-index mapping
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print(f"\nVocabulary: {vocabulary}")
print(f"Vocabulary size: {vocab_size}")
print(f"\nWord to Index mapping:")
for word, idx in word_to_idx.items():
    print(f"  '{word}' -> {idx}")

# ============================================================================
# PART 2: Creating One-Hot Vectors
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Creating One-Hot Vectors")
print("=" * 70)

def one_hot_encode(word, word_to_idx, vocab_size):
    """Convert a word to one-hot vector"""
    vector = np.zeros(vocab_size)
    if word in word_to_idx:
        vector[word_to_idx[word]] = 1
    return vector

# Encode each word
print("\nOne-hot vectors for each word:")
for word in vocabulary:
    vec = one_hot_encode(word, word_to_idx, vocab_size)
    print(f"  '{word}': {vec.astype(int)}")

# Visualize as heatmap
one_hot_matrix = np.array([one_hot_encode(w, word_to_idx, vocab_size) for w in vocabulary])

plt.figure(figsize=(10, 4))
plt.imshow(one_hot_matrix, cmap='Blues', aspect='auto')
plt.colorbar(label='Value')
plt.yticks(range(vocab_size), vocabulary)
plt.xticks(range(vocab_size), range(vocab_size))
plt.xlabel('Vector Position')
plt.ylabel('Word')
plt.title('One-Hot Encoding Matrix')
plt.tight_layout()
plt.savefig('one_hot_visualization.png', dpi=150)
print("\n Visualization saved to: one_hot_visualization.png")

# ============================================================================
# PART 3: One-Hot Encoding for Sentences
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Encoding Sentences")
print("=" * 70)

def encode_sentence(sentence, word_to_idx, vocab_size):
    """Encode a sentence as matrix of one-hot vectors"""
    words = sentence.lower().split()
    matrix = []
    for word in words:
        vec = one_hot_encode(word, word_to_idx, vocab_size)
        matrix.append(vec)
    return np.array(matrix), words

sentence = "cat dog bird"
encoded, words = encode_sentence(sentence, word_to_idx, vocab_size)

print(f"\nSentence: '{sentence}'")
print(f"Encoded shape: {encoded.shape}")
print(f"\nEncoding matrix:")
print(encoded.astype(int))

# Bag of Words representation
bow = np.sum(encoded, axis=0)
print(f"\nBag of Words (sum): {bow.astype(int)}")
print("(Loses word order but captures word presence)")

# ============================================================================
# PART 4: The Dimensionality Problem
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: The Dimensionality Problem")
print("=" * 70)

print("\nReal-world vocabulary sizes:")
print("-" * 40)

vocab_sizes = {
    "Our toy example": 5,
    "Small dataset": 5_000,
    "Medium dataset": 20_000,
    "Large dataset (news)": 50_000,
    "Very large (Wikipedia)": 100_000,
    "BERT vocabulary": 30_522
}

for name, size in vocab_sizes.items():
    memory_per_word = size * 4  # 4 bytes per float32
    memory_kb = memory_per_word / 1024
    print(f"  {name:25s}: {size:>7,} dims -> {memory_kb:>7.1f} KB per word")

print("\n Problem: Vocabulary of 50,000 means each word is a 50,000-dim vector!")
print("For a document with 1000 words: 50,000 x 1000 = 50 million elements!")

# ============================================================================
# PART 5: The Sparsity Problem
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: The Sparsity Problem")
print("=" * 70)

# Calculate sparsity
for vocab_size in [100, 1000, 10000, 50000]:
    nonzero = 1
    total = vocab_size
    sparsity = (total - nonzero) / total * 100
    print(f"Vocab size {vocab_size:>6,}: {sparsity:.4f}% zeros")

print("\n 99.99% of your data is zeros!")
print("This wastes memory and computation.")

# ============================================================================
# PART 6: The Semantic Similarity Problem
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: No Semantic Similarity")
print("=" * 70)

print("\nCalculating cosine similarity between words...")

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# Get one-hot vectors
cat_vec = one_hot_encode("cat", word_to_idx, vocab_size)
dog_vec = one_hot_encode("dog", word_to_idx, vocab_size)
bird_vec = one_hot_encode("bird", word_to_idx, vocab_size)
fish_vec = one_hot_encode("fish", word_to_idx, vocab_size)

# Calculate all pairwise similarities
words = ["cat", "dog", "bird", "fish"]
vectors = [cat_vec, dog_vec, bird_vec, fish_vec]

print("\nCosine similarity matrix:")
print("\n" + " " * 8, end="")
for w in words:
    print(f"{w:>8}", end="")
print()

similarity_matrix = np.zeros((len(words), len(words)))
for i, (w1, v1) in enumerate(zip(words, vectors)):
    print(f"{w1:>8}", end="")
    for j, (w2, v2) in enumerate(zip(words, vectors)):
        sim = cosine_similarity(v1, v2)
        similarity_matrix[i, j] = sim
        print(f"{sim:>8.2f}", end="")
    print()

print("\n KEY INSIGHT:")
print("- cat vs dog: 0.00 (but both are animals!)")
print("- cat vs fish: 0.00 (both are animals!)")
print("- cat vs cat: 1.00 (only self-similarity)")
print("\nOne-hot encoding treats ALL pairs as equally dissimilar!")
print("This is a fundamental limitation.")

# Visualize similarity matrix
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.title('One-Hot Encoding: Cosine Similarity Matrix\n(All non-identical pairs have 0 similarity)')
for i in range(len(words)):
    for j in range(len(words)):
        plt.text(j, i, f'{similarity_matrix[i,j]:.2f}', ha='center', va='center', fontsize=12)
plt.tight_layout()
plt.savefig('one_hot_similarity.png', dpi=150)
print("\n Similarity visualization saved to: one_hot_similarity.png")

# ============================================================================
# PART 7: When One-Hot IS Appropriate
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: When to Use One-Hot Encoding")
print("=" * 70)

print("\n One-hot encoding IS appropriate for:")
print("-" * 40)

print("""
1. OUTPUT LABELS in classification:
   - Predicting one of 10 classes
   - One-hot is the target format for softmax
   
   Example:
   y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # Class 3
   y_pred = [0.01, 0.02, 0.05, 0.85, 0.02, ...]  # Softmax output

2. SMALL, FIXED CATEGORICAL features:
   - Days of week (7 categories)
   - Colors (limited set)
   - Country codes (fixed set)
   
3. Traditional ML models:
   - Logistic regression
   - Decision trees
   - When using sparse matrix support
""")

# Example: One-hot for labels
from tensorflow.keras.utils import to_categorical

labels = [3, 1, 4, 0, 2]
one_hot_labels = to_categorical(labels, num_classes=5)

print("\nExample: Classification labels")
print(f"Integer labels: {labels}")
print(f"One-hot labels:\n{one_hot_labels.astype(int)}")

# ============================================================================
# PART 8: Preview - Why We Need Embeddings
# ============================================================================

print("\n" + "=" * 70)
print("PART 8: Preview - Why We Need Dense Embeddings")
print("=" * 70)

print("\nComparing One-Hot vs Dense Embeddings:")
print("-" * 50)

print("""
                  One-Hot          Dense Embeddings
                  --------         ----------------
Dimensions:       10,000+          100-300
Sparsity:         99.99% zeros     All non-zero
cat-dog sim:      0.0              ~0.8 (both animals!)
Memory:           Huge             Reasonable
Semantics:        None             Captures meaning
OOV handling:     Poor             Better (with subwords)
""")

print("\nNEXT DEMO: Introduction to Embeddings")
print("We'll see how 'king - man + woman = queen' works!")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: One-Hot Encoding")
print("=" * 70)

print("\nKey Takeaways:")
print("1. One-hot: Each word = sparse vector with single 1")
print("2. Dimensionality equals vocabulary size (10,000+)")
print("3. 99.99% of values are zeros (extremely sparse)")
print("4. ALL word pairs are equally dissimilar (no semantics)")
print("5. Still useful for output labels and small categories")
print("6. Dense embeddings solve these problems (next topic)")

print("\n" + "=" * 70)

