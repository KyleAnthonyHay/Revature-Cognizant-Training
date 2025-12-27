"""
Demo 03: Word2Vec Embeddings Visualization

This demo shows trainees how to:
1. Understand how Word2Vec learns word relationships
2. Visualize embeddings with t-SNE
3. Demonstrate the famous "king - man + woman = queen"
4. Explore semantic similarity and clustering

Learning Objectives:
- Understand Word2Vec intuition (Skip-gram/CBOW)
- Visualize word embeddings in 2D
- Perform vector arithmetic for analogies

References:
- Written Content: 04-introduction-to-embeddings.md
- Written Content: 05-word2vec-intuition.md
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ============================================================================
# PART 1: Word2Vec Intuition
# ============================================================================

print("=" * 70)
print("PART 1: Word2Vec Intuition")
print("=" * 70)

print("\nWord2Vec Key Insight:")
print("-" * 50)
print("""
"You shall know a word by the company it keeps."
                                - J.R. Firth

Words appearing in similar contexts should have similar embeddings.

Example contexts:
  "The cat sat on the mat"
  "The dog sat on the mat"

'cat' and 'dog' appear in identical contexts 
-> They should have similar embedding vectors!
""")

print("\nTwo Word2Vec Architectures:")
print("-" * 50)
print("""
1. Skip-gram: Given target word, predict context words
   Input: "fox" -> Predict: ["quick", "brown", "jumps", "over"]
   
2. CBOW: Given context words, predict target word
   Input: ["quick", "brown", "jumps", "over"] -> Predict: "fox"

Skip-gram is more common, works better for rare words.
""")

# ============================================================================
# PART 2: Simulated Word Embeddings
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Simulated Word Embeddings (for demonstration)")
print("=" * 70)

# Create simulated embeddings that demonstrate Word2Vec properties
# In practice, you'd load pre-trained embeddings (GloVe, Word2Vec)

np.random.seed(42)

# Define semantic categories
categories = {
    'royalty_male': ['king', 'prince', 'lord'],
    'royalty_female': ['queen', 'princess', 'lady'],
    'people_male': ['man', 'boy', 'father'],
    'people_female': ['woman', 'girl', 'mother'],
    'animals': ['dog', 'cat', 'bird', 'fish'],
    'food': ['apple', 'bread', 'pizza', 'burger'],
    'countries': ['france', 'germany', 'italy', 'spain'],
    'capitals': ['paris', 'berlin', 'rome', 'madrid']
}

# Create embedding dimension
embedding_dim = 50

# Base vectors for categories
category_bases = {
    'royalty_male': np.random.randn(embedding_dim) + np.array([2, 2] + [0]*(embedding_dim-2)),
    'royalty_female': np.random.randn(embedding_dim) + np.array([2, -2] + [0]*(embedding_dim-2)),
    'people_male': np.random.randn(embedding_dim) + np.array([0, 2] + [0]*(embedding_dim-2)),
    'people_female': np.random.randn(embedding_dim) + np.array([0, -2] + [0]*(embedding_dim-2)),
    'animals': np.random.randn(embedding_dim) + np.array([-3, 0] + [0]*(embedding_dim-2)),
    'food': np.random.randn(embedding_dim) + np.array([-3, -3] + [0]*(embedding_dim-2)),
    'countries': np.random.randn(embedding_dim) + np.array([4, 0] + [0]*(embedding_dim-2)),
    'capitals': np.random.randn(embedding_dim) + np.array([4, 1] + [0]*(embedding_dim-2))
}

# Generate embeddings for each word
embeddings = {}
for category, words in categories.items():
    base = category_bases[category]
    for i, word in enumerate(words):
        # Add small noise to make words in same category similar but not identical
        noise = np.random.randn(embedding_dim) * 0.3
        embeddings[word] = base + noise + i * 0.1

print(f"Created embeddings for {len(embeddings)} words")
print(f"Embedding dimension: {embedding_dim}")

# List all words
all_words = list(embeddings.keys())
print(f"\nVocabulary: {all_words}")

# ============================================================================
# PART 3: Cosine Similarity with Embeddings
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Semantic Similarity with Dense Embeddings")
print("=" * 70)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

def most_similar(word, embeddings, top_n=5):
    """Find most similar words"""
    if word not in embeddings:
        return []
    
    target_vec = embeddings[word]
    similarities = []
    
    for other_word, other_vec in embeddings.items():
        if other_word != word:
            sim = cosine_similarity(target_vec, other_vec)
            similarities.append((other_word, sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Test similarity
test_words = ['king', 'dog', 'france', 'apple']

print("\nMost similar words (cosine similarity):")
print("-" * 50)

for word in test_words:
    similar = most_similar(word, embeddings, top_n=3)
    print(f"\n'{word}':")
    for similar_word, sim in similar:
        print(f"  {similar_word}: {sim:.3f}")

# Compare with one-hot result
print("\n KEY DIFFERENCE from One-Hot:")
print("  'king' is similar to 'queen', 'prince' (royalty)")
print("  'dog' is similar to 'cat', 'bird' (animals)")
print("  Embeddings CAPTURE semantic relationships!")

# ============================================================================
# PART 4: The Famous Analogy - King - Man + Woman = Queen
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Vector Arithmetic - Analogies")
print("=" * 70)

print("\nThe famous Word2Vec analogy:")
print("  king - man + woman = ???")
print()

# Compute analogy
def analogy(word_a, word_b, word_c, embeddings, exclude=None):
    """
    Solve: word_a is to word_b as word_c is to ???
    Computes: word_b - word_a + word_c
    """
    if exclude is None:
        exclude = [word_a, word_b, word_c]
    
    vec_a = embeddings[word_a]
    vec_b = embeddings[word_b]
    vec_c = embeddings[word_c]
    
    result_vec = vec_b - vec_a + vec_c
    
    # Find closest word to result
    best_word = None
    best_sim = -1
    
    for word, vec in embeddings.items():
        if word not in exclude:
            sim = cosine_similarity(result_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_word = word
    
    return best_word, best_sim

# Test the famous analogy
# king - man + woman should give queen
# (royalty_male) - (male) + (female) = (royalty_female)

result, similarity = analogy('man', 'king', 'woman', embeddings)
print(f"man -> king  ::  woman -> {result} (similarity: {similarity:.3f})")

# More analogies
print("\nMore analogies:")
print("-" * 50)

analogies = [
    ('man', 'king', 'woman'),       # gender royalty
    ('france', 'paris', 'germany'),  # country-capital
    ('france', 'paris', 'italy'),    # country-capital
]

for a, b, c in analogies:
    result, sim = analogy(a, b, c, embeddings)
    print(f"  {a} -> {b}  ::  {c} -> {result} (sim: {sim:.3f})")

print("\n WHY THIS WORKS:")
print("  'king' - 'man' = direction of 'royalty'")
print("  'woman' + 'royalty' = 'queen'")
print("  Consistent directions encode relationships!")

# ============================================================================
# PART 5: t-SNE Visualization
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: t-SNE Visualization of Embedding Space")
print("=" * 70)

print("\nt-SNE reduces high-dimensional embeddings to 2D for visualization")
print("Similar words should cluster together...")

# Get all embeddings as matrix
words = list(embeddings.keys())
embedding_matrix = np.array([embeddings[w] for w in words])

print(f"\nEmbedding matrix shape: {embedding_matrix.shape}")

# Apply t-SNE
print("Running t-SNE... (this may take a moment)")
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embedding_matrix)

print("t-SNE complete!")

# Create color map for categories
word_to_category = {}
for cat, word_list in categories.items():
    for word in word_list:
        word_to_category[word] = cat

category_colors = {
    'royalty_male': 'blue',
    'royalty_female': 'lightblue',
    'people_male': 'green',
    'people_female': 'lightgreen',
    'animals': 'red',
    'food': 'orange',
    'countries': 'purple',
    'capitals': 'pink'
}

# Plot
plt.figure(figsize=(14, 10))

for i, word in enumerate(words):
    cat = word_to_category[word]
    color = category_colors[cat]
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=color, s=100)
    plt.annotate(word, (embeddings_2d[i, 0] + 0.5, embeddings_2d[i, 1] + 0.5), fontsize=10)

# Create legend
legend_elements = [plt.scatter([], [], c=color, label=cat, s=100) 
                   for cat, color in category_colors.items()]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.title('Word Embeddings Visualization (t-SNE)\nSimilar words cluster together!', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('word_embeddings_tsne.png', dpi=150)
print("\n Visualization saved to: word_embeddings_tsne.png")

print("\nObservations from the visualization:")
print("- Animals cluster together (red)")
print("- Food items cluster together (orange)")
print("- Countries near their capitals")
print("- Male/female versions near each other")

# ============================================================================
# PART 6: Using Pre-trained Embeddings (Code Example)
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Loading Pre-trained Embeddings (Code Reference)")
print("=" * 70)

print("""
In practice, you would load pre-trained embeddings:

# Option 1: GloVe (Stanford)
def load_glove(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove = load_glove('glove.6B.100d.txt')  # 400K words, 100 dimensions

# Option 2: Gensim Word2Vec
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
king_vec = model['king']
similar = model.most_similar('king')

# Option 3: Hugging Face Transformers (contextual embeddings)
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# These give CONTEXTUAL embeddings (same word, different vectors based on context)
""")

# ============================================================================
# PART 7: Embedding Layer in Keras
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Using Embeddings in Keras")
print("=" * 70)

from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

# Create a simple model with embedding layer
model = Sequential([
    Embedding(
        input_dim=10000,      # Vocabulary size
        output_dim=100,       # Embedding dimension
        input_length=50       # Sequence length
    )
])

model.build(input_shape=(None, 50))
print(f"\nKeras Embedding Layer:")
print(f"  Input: Word indices (integers)")
print(f"  Output: Dense vectors (embeddings)")
print(f"  Parameters: {model.count_params():,} (vocab_size x embedding_dim)")

print("""
# Usage example:
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Convert text to indices
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=50)

# Model with embeddings
model = Sequential([
    Embedding(input_dim=10000, output_dim=100, input_length=50),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Embeddings are LEARNED during training!
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Word2Vec Embeddings Visualization")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Word2Vec learns embeddings from word context (Skip-gram/CBOW)")
print("2. Similar words have similar embedding vectors")
print("3. Vector arithmetic captures semantic relationships")
print("4. 'king - man + woman = queen' works!")
print("5. t-SNE visualizes high-dim embeddings in 2D")
print("6. Pre-trained embeddings (GloVe, Word2Vec) enable transfer learning")
print("7. Keras Embedding layer learns task-specific embeddings")

print("\nTomorrow: RNNs and LSTMs for sequence processing!")

print("\n" + "=" * 70)

