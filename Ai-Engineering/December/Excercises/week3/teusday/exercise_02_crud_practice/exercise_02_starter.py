"""
Exercise 02: CRUD Practice - Starter Code

Complete the TODOs to practice all CRUD operations in Chroma DB.

Prerequisites:
- pip install chromadb

Hints:
- Reading 04 (crud-operations-vector-data.md) has method signatures
- Demo 02 (demo_02_crud_operations.py) shows each operation
"""

import chromadb

# ============================================================================
# SETUP
# ============================================================================

print("=" * 60)
print("Exercise 02: CRUD Practice")
print("=" * 60)

# Create an in-memory client for practice
client = chromadb.Client()

# Create a collection for our knowledge base
# Hint: client.create_collection() or client.get_or_create_collection()
collection = client.create_collection("knowledge_base")


# ============================================================================
# PART 1: CREATE Operations
# ============================================================================

print("\n" + "=" * 60)
print("Part 1: CREATE Operations")
print("=" * 60)

# TODO 1.1: Add these 5 documents with metadata
# Hint: collection.add(documents=[...], ids=[...], metadatas=[...])

documents = [
    "Python is a versatile programming language used for web development, data science, and automation.",
    "Machine learning models require large datasets for training and validation.",
    "Docker containers provide consistent environments across development and production.",
    "Neural networks are inspired by the structure of biological brains.",
    "REST APIs use HTTP methods like GET, POST, PUT, and DELETE."
]

ids = ["doc_python", "doc_ml", "doc_docker", "doc_neural", "doc_api"]

metadatas = [
    {"category": "programming", "author": "alice", "difficulty": "beginner"},
    {"category": "ai", "author": "bob", "difficulty": "intermediate"},
    {"category": "devops", "author": "alice", "difficulty": "intermediate"},
    {"category": "ai", "author": "carol", "difficulty": "advanced"},
    {"category": "programming", "author": "bob", "difficulty": "beginner"}
]

# Your add() call here:
# collection.add(...)

print(f"Added documents. Collection count: {collection.count()}")


# TODO 1.2: Try adding a duplicate ID and observe the error
# Hint: Wrap in try/except to see the error message
print("\n[Task 1.2] Testing duplicate ID handling...")
# Your code here:


# ============================================================================
# PART 2: READ Operations
# ============================================================================

print("\n" + "=" * 60)
print("Part 2: READ Operations")
print("=" * 60)

# TODO 2.1: Get document by ID
# Hint: collection.get(ids=["doc_python"])
print("\n[Task 2.1] Get by ID 'doc_python':")
# result = collection.get(...)
# print(result)


# TODO 2.2a: Get documents where category = "ai"
# Hint: collection.get(where={"category": "ai"})
print("\n[Task 2.2a] Get documents in 'ai' category:")
# Your code here:


# TODO 2.2b: Get documents where author = "alice"
print("\n[Task 2.2b] Get documents by author 'alice':")
# Your code here:


# TODO 2.3: Similarity query
# Hint: collection.query(query_texts=[...], n_results=3)
print("\n[Task 2.3] Query: 'How do I build an AI model?'")
# results = collection.query(...)
# for doc, dist in zip(results['documents'][0], results['distances'][0]):
#     print(f"  [{dist:.4f}] {doc[:50]}...")


# TODO 2.4: Query with filter
# Hint: Combine query_texts with where parameter
print("\n[Task 2.4] Query programming docs for 'build a web service':")
# Your code here:


# ============================================================================
# PART 3: UPDATE Operations
# ============================================================================

print("\n" + "=" * 60)
print("Part 3: UPDATE Operations")
print("=" * 60)

# TODO 3.1: Update metadata for doc_ml
# Hint: collection.update(ids=[...], metadatas=[...])
print("\n[Task 3.1] Update 'doc_ml' difficulty to 'advanced':")
# Your code here:
# Verify: print(collection.get(ids=["doc_ml"]))


# TODO 3.2: Update document content for doc_api
# Hint: This triggers re-embedding!
print("\n[Task 3.2] Update 'doc_api' content:")
new_api_text = "REST APIs use HTTP methods like GET, POST, PUT, and DELETE to perform CRUD operations on resources."
# Your code here:


# TODO 3.3: Use upsert for doc_python (update) and doc_cloud (insert)
# Hint: collection.upsert() works for both
print("\n[Task 3.3] Upsert operations:")
# Your code here:


# ============================================================================
# PART 4: DELETE Operations
# ============================================================================

print("\n" + "=" * 60)
print("Part 4: DELETE Operations")
print("=" * 60)

# TODO 4.1: Delete by ID
# Hint: collection.delete(ids=[...])
print("\n[Task 4.1] Delete 'doc_docker':")
print(f"Count before: {collection.count()}")
# Your code here:
print(f"Count after: {collection.count()}")


# TODO 4.2: Delete by filter
# Hint: collection.delete(where={...})
print("\n[Task 4.2] Delete all documents by 'carol':")
# Your code here:
print(f"Count after: {collection.count()}")


# TODO 4.3: Verify deletions
print("\n[Task 4.3] Verification:")
# Try to get deleted documents - should return empty


# ============================================================================
# PART 5: Collection Management
# ============================================================================

print("\n" + "=" * 60)
print("Part 5: Collection Management")
print("=" * 60)

# TODO 5.1: List all collections
# Hint: client.list_collections()
print("\n[Task 5.1] List collections:")
# Your code here:


# TODO 5.2: Create a second collection
print("\n[Task 5.2] Create 'archived_docs' collection:")
# Your code here:


# TODO 5.3: Delete a collection
# Hint: client.delete_collection(name=...)
print("\n[Task 5.3] Delete 'archived_docs' collection:")
# Your code here:


print("\n" + "=" * 60)
print("Exercise Complete!")
print("=" * 60)
