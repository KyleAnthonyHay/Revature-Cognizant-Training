"""
Exercise 02: Embedding Exploration - Starter Code

Complete the TODOs to explore text embeddings and similarity.

Prerequisites:
- pip install numpy sentence-transformers

Hints:
- Demo 02 (demo_02_embedding_generation.py) shows model loading
- Reading 05 (vector-similarity-concepts.md) has the cosine formula
"""

import numpy as np
# TODO 1.1: Import SentenceTransformer from sentence_transformers


# ============================================================================
# PART 1: Load the Embedding Model
# ============================================================================

print("=" * 60)
print("Part 1: Loading the Embedding Model")
print("=" * 60)

# TODO 1.1: Load the 'all-MiniLM-L6-v2' model
# Hint: model = SentenceTransformer('...')
model = None  # Replace with actual model loading

# TODO 1.2: Print the embedding dimension
# Hint: Look for a method with "embedding" and "dimension" in the name
print(f"Embedding dimension: ???")


# ============================================================================
# PART 2: Generate Embeddings
# ============================================================================

print("\n" + "=" * 60)
print("Part 2: Generate Embeddings")
print("=" * 60)

# TODO 2.1: Encode a single sentence
single_sentence = "Machine learning is transforming industries"
# Hint: embedding = model.encode(...)
single_embedding = None  # Replace

# Print embedding properties
print(f"Shape: {single_embedding.shape if single_embedding is not None else '???'}")
print(f"First 5 values: ???")
print(f"Min: ???, Max: ???")


# TODO 2.2: Batch encode these sentences
sentences = [
    "The cat sat on the mat",
    "A kitten rested on the rug",
    "Dogs are loyal companions",
    "Python is a programming language",
    "The python snake is quite long",
    "I love coding in Python"
]

# Hint: Pass the list directly to model.encode()
embeddings = None  # Replace

print(f"\nBatch shape: {embeddings.shape if embeddings is not None else '???'}")


# ============================================================================
# PART 3: Calculate Similarities
# ============================================================================

print("\n" + "=" * 60)
print("Part 3: Calculate Similarities")
print("=" * 60)

# TODO 3.1: Implement cosine similarity
def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Hint: Use np.dot() for dot product
    Hint: Use np.linalg.norm() for magnitude
    """
    # Your implementation here
    pass


# TODO 3.2: Build similarity matrix
# For each pair of sentences, calculate similarity
print("\nSimilarity Matrix:")
print("-" * 40)

# Example structure (fill in the actual calculations):
# for i in range(len(sentences)):
#     for j in range(len(sentences)):
#         sim = cosine_similarity(embeddings[i], embeddings[j])
#         ...


# ============================================================================
# PART 3.3: Analysis Questions
# ============================================================================

print("\n" + "=" * 60)
print("Part 3.3: Analysis")
print("=" * 60)

# TODO: Answer these questions based on your similarity matrix

print("""
Q1: Which two sentences have the highest similarity (besides identical)?
    Answer: 

Q2: How similar are 'Python is a programming language' and 'The python snake is quite long'?
    Similarity score: 
    Interpretation: 

Q3: Which sentence is most 'isolated' (lowest average similarity)?
    Answer: 
""")


# ============================================================================
# PART 4: Semantic Clustering
# ============================================================================

print("\n" + "=" * 60)
print("Part 4: Semantic Clustering")
print("=" * 60)

# TODO 4.1: Group sentences by topic
print("""
Cluster A (Animal-related):
    - 

Cluster B (Programming-related):
    - 

Outliers:
    - 
""")

# TODO 4.2: What similarity threshold would you use?
print("""
Recommended threshold for 'related' sentences: ???
Justification:
""")


print("\n" + "=" * 60)
print("Exercise Complete!")
print("=" * 60)
