# Exercise 03: Embedding Explorer

## Learning Objectives

- Explore semantic relationships in word embeddings
- Perform vector arithmetic (king - man + woman = queen)
- Visualize embedding spaces using dimensionality reduction
- Integrate Keras Embedding layers into models

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_03_word2vec_visualization.py`, you saw the famous analogy: king - man + woman = queen. In this exercise, you'll explore embedding properties and build models with embedding layers.

## Your Challenge

Using pre-trained or simulated embeddings, explore semantic relationships and build an embedding-based classifier.

---

## Part 1: Load or Create Embeddings (15 min)

### Task 1.1: Create Rich Semantic Embeddings

For this exercise, we'll create embeddings that demonstrate Word2Vec-like properties.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Create embeddings with structured semantic relationships
def create_rich_embeddings(dim=50):
    """
    Create embeddings with clear semantic structure.
    
    Categories:
    - Royalty: king, queen, prince, princess
    - Gender: man, woman, boy, girl
    - Countries/Capitals: france, paris, germany, berlin, italy, rome
    - Actions: walk, run, swim, fly
    """
    np.random.seed(42)
    embeddings = {}
    
    # Create semantic axes
    royalty_axis = np.random.randn(dim)
    gender_axis = np.random.randn(dim)
    country_axis = np.random.randn(dim)
    capital_axis = np.random.randn(dim)
    
    # Royalty with gender
    embeddings["king"] = royalty_axis + gender_axis * -1  # male royalty
    embeddings["queen"] = royalty_axis + gender_axis * 1  # female royalty
    embeddings["prince"] = royalty_axis * 0.7 + gender_axis * -1  # young male royalty
    embeddings["princess"] = royalty_axis * 0.7 + gender_axis * 1  # young female royalty
    
    # Gender without royalty
    embeddings["man"] = gender_axis * -1
    embeddings["woman"] = gender_axis * 1
    embeddings["boy"] = gender_axis * -1 * 0.7
    embeddings["girl"] = gender_axis * 1 * 0.7
    
    # Countries and capitals
    embeddings["france"] = country_axis + np.random.randn(dim) * 0.3
    embeddings["paris"] = embeddings["france"] + capital_axis
    embeddings["germany"] = country_axis + np.random.randn(dim) * 0.3
    embeddings["berlin"] = embeddings["germany"] + capital_axis
    embeddings["italy"] = country_axis + np.random.randn(dim) * 0.3
    embeddings["rome"] = embeddings["italy"] + capital_axis
    
    # Normalize all embeddings
    for word in embeddings:
        embeddings[word] = embeddings[word] / np.linalg.norm(embeddings[word])
    
    return embeddings
```

---

## Part 2: Semantic Similarity Analysis (15 min)

### Task 2.1: Find Similar Words

Implement a function to find the most similar words to a query:

```python
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def most_similar(word, embeddings, top_n=5):
    """
    Find the most similar words to the query word.
    
    Returns:
        List of (word, similarity) tuples
    """
    # TODO: Implement
    # 1. Get query embedding
    # 2. Compute similarity to all other words
    # 3. Sort by similarity descending
    # 4. Return top_n results (excluding query word)
    pass

# Test
embeddings = create_rich_embeddings()
print("Most similar to 'king':")
for word, sim in most_similar("king", embeddings):
    print(f"  {word}: {sim:.4f}")
```

### Task 2.2: Similarity Matrix

Create a heatmap of similarities between word groups:

```python
def plot_similarity_matrix(words, embeddings):
    """
    Create heatmap of pairwise similarities.
    """
    n = len(words)
    matrix = np.zeros((n, n))
    
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            matrix[i, j] = cosine_similarity(embeddings[w1], embeddings[w2])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(n), words, rotation=45, ha='right')
    plt.yticks(range(n), words)
    plt.title('Word Embedding Similarities')
    plt.tight_layout()
    plt.show()

# Visualize royalty and gender words
test_words = ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl"]
plot_similarity_matrix(test_words, embeddings)
```

---

## Part 3: Analogy Solving (15 min)

### Task 3.1: Implement Analogy Function

The famous analogy: king - man + woman = ?

```python
def solve_analogy(a, b, c, embeddings, top_n=3):
    """
    Solve: a is to b as c is to ?
    
    Method: result = c + (b - a)
    
    Example: king is to queen as man is to ?
             man + (queen - king) = woman
    
    Returns:
        List of (word, similarity) tuples
    """
    # TODO: Implement
    # 1. Compute target vector: embeddings[c] + (embeddings[b] - embeddings[a])
    # 2. Find words most similar to target
    # 3. Exclude a, b, c from results
    pass

# Test analogies
print("king : queen :: man : ?")
print(solve_analogy("king", "queen", "man", embeddings))

print("\nfrance : paris :: germany : ?")
print(solve_analogy("france", "paris", "germany", embeddings))

print("\nman : boy :: woman : ?")
print(solve_analogy("man", "boy", "woman", embeddings))
```

### Task 3.2: Create Your Own Analogies

Test at least 3 additional analogies using the embeddings. Document which work and which don't.

---

## Part 4: Embedding Visualization (10 min)

### Task 4.1: t-SNE Projection

Visualize the embedding space in 2D:

```python
def visualize_embeddings(embeddings, categories=None):
    """
    Use t-SNE to project embeddings to 2D and visualize.
    
    Args:
        embeddings: Dict of word -> vector
        categories: Optional dict of word -> category for coloring
    """
    words = list(embeddings.keys())
    vectors = np.array([embeddings[w] for w in words])
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1))
    coords = tsne.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    if categories:
        unique_cats = list(set(categories.values()))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
        color_map = dict(zip(unique_cats, colors))
        
        for i, word in enumerate(words):
            cat = categories.get(word, "other")
            color = color_map.get(cat, "gray")
            plt.scatter(coords[i, 0], coords[i, 1], c=[color], s=100)
            plt.annotate(word, (coords[i, 0]+0.5, coords[i, 1]+0.5))
    else:
        plt.scatter(coords[:, 0], coords[:, 1], s=100)
        for i, word in enumerate(words):
            plt.annotate(word, (coords[i, 0]+0.5, coords[i, 1]+0.5))
    
    plt.title('Word Embeddings (t-SNE Projection)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()

# Define categories
categories = {
    "king": "royalty", "queen": "royalty", "prince": "royalty", "princess": "royalty",
    "man": "gender", "woman": "gender", "boy": "gender", "girl": "gender",
    "france": "country", "paris": "capital", "germany": "country", "berlin": "capital",
    "italy": "country", "rome": "capital"
}

visualize_embeddings(embeddings, categories)
```

---

## Part 5: Keras Embedding Layer (5 min)

### Task 5.1: Build Model with Embedding

Integrate embeddings into a Keras model:

```python
from tensorflow import keras
from tensorflow.keras import layers

def build_embedding_classifier(vocab_size, embedding_dim, max_length, num_classes):
    """
    Build a text classifier using Keras Embedding layer.
    """
    model = keras.Sequential([
        # Embedding layer: converts word indices to dense vectors
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),
        # Flatten embeddings for classification
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example usage
model = build_embedding_classifier(
    vocab_size=10000,
    embedding_dim=100,
    max_length=50,
    num_classes=2
)
model.summary()
```

---

## Reflection Questions

1. **Why does king - man + woman = queen work mathematically?** What does this tell us about how embeddings encode meaning?

2. **Looking at the t-SNE plot, which categories cluster together?** Any surprising relationships?

3. **What would happen if you used random embeddings instead of semantic ones?** Would analogies still work?

4. **How does the Keras Embedding layer learn good embeddings?** What signal does it use during training?

---

## Definition of Done

- [ ] Created rich semantic embeddings
- [ ] Implemented most_similar function
- [ ] Created similarity matrix heatmap
- [ ] Solved at least 5 analogies (3 given + 2 your own)
- [ ] Visualized embeddings with t-SNE
- [ ] Built Keras model with Embedding layer
- [ ] Reflection questions answered

---

## Bonus Challenge

Load pre-trained GloVe embeddings and test if real embeddings work better:

```python
def load_glove(path, dim=50):
    """Load GloVe embeddings from file."""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Download from: https://nlp.stanford.edu/projects/glove/
# glove = load_glove('glove.6B.50d.txt')
```

---

## Solution

The solution is available in `solutions/exercise_03_solution.py`. Only check after completing your attempt!

