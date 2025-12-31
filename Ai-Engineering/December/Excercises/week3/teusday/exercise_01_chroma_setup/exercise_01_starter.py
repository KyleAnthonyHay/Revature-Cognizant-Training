"""
Exercise 01: Chroma DB Setup - Starter Code

Verify your Chroma DB installation and create your first collection.

Prerequisites:
- pip install chromadb

Hints:
- Reading 02 has installation steps
- Demo 01 shows Client() vs PersistentClient() patterns
"""

import sys

# ============================================================================
# PART 1: Installation Verification
# ============================================================================

print("=" * 60)
print("Exercise 01: Chroma DB Setup")
print("=" * 60)

print("\n--- PART 1: INSTALLATION CHECK ---")

# TODO 1.1: Import chromadb
# Hint: If this fails, run: pip install chromadb
try:
    import chromadb
    print(f"[OK] chromadb imported successfully")
    print(f"  Version: {chromadb.__version__}")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("  Run: pip install chromadb")
    sys.exit(1)

# Check sqlite3
import sqlite3
print(f"[OK] sqlite3 version: {sqlite3.sqlite_version}")


# ============================================================================
# PART 2: Create Your First Client
# ============================================================================

print("\n--- PART 2: CREATE CLIENTS ---")

# TODO 2.1: Create an in-memory client
# Hint: chromadb.Client() creates an ephemeral client
# client = chromadb.???()
print("\nTODO: Create in-memory client")
# print(f"Heartbeat: {client.heartbeat()}")


# TODO 2.2: Create a persistent client
# Hint: chromadb.PersistentClient(path="./my_chroma_db")
# persistent_client = chromadb.???("./my_chroma_db")
print("\nTODO: Create persistent client at './my_chroma_db'")


# ============================================================================
# PART 3: Hello World Vector
# ============================================================================

print("\n--- PART 3: HELLO WORLD VECTOR ---")

# TODO 3.1: Create a collection
# Hint: client.create_collection("hello_vectors")
# collection = client.???("hello_vectors")
print("\nTODO: Create 'hello_vectors' collection")


# TODO 3.2: Add your first document
# Hint: collection.add(documents=["..."], ids=["..."])
# Remember: documents and ids are both LISTS!
print("\nTODO: Add document 'Hello, vector database world!' with id 'hello_1'")


# TODO 3.3: Query it back
# Hint: collection.query(query_texts=["..."], n_results=1)
print("\nTODO: Query for 'greeting' and print result")


# ============================================================================
# VERIFICATION SCRIPT
# ============================================================================

print("\n" + "=" * 60)
print("VERIFICATION SCRIPT")
print("=" * 60)

print("""
Run this code to verify everything works:

import chromadb

# Test 1: In-memory client
client = chromadb.Client()
print(f"[OK] In-memory client, heartbeat: {client.heartbeat()}")

# Test 2: Create collection
collection = client.create_collection("test_collection")
print(f"[OK] Collection created: {collection.name}")

# Test 3: Add document
collection.add(documents=["test document"], ids=["test_1"])
print(f"[OK] Document added, count: {collection.count()}")

# Test 4: Query
results = collection.query(query_texts=["test"], n_results=1)
print(f"[OK] Query successful, found: {results['documents'][0][0]}")

print("\\n All tests passed!")
""")


print("\n" + "=" * 60)
print("Complete the TODOs above, then run the verification script!")
print("=" * 60)
