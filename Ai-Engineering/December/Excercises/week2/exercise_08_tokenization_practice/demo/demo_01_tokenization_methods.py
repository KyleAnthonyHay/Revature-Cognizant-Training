"""
Demo 01: Tokenization Methods

This demo shows trainees how to:
1. Understand different tokenization approaches
2. Compare word-level, character-level, and subword tokenization
3. Use NLTK, spaCy, and Hugging Face tokenizers
4. Build vocabulary from corpus

Learning Objectives:
- Understand why tokenization is the first step in NLP
- Compare different tokenization strategies
- Learn to handle edge cases (punctuation, contractions)

References:
- Written Content: 02-tokenization-text-processing.md
"""

import re
import numpy as np
from collections import Counter

# ============================================================================
# PART 1: Why Tokenization?
# ============================================================================

print("=" * 70)
print("PART 1: Introduction to Tokenization")
print("=" * 70)

print("\nTokenization: Breaking text into smaller units (tokens)")
print("-" * 50)

sample_text = "I love natural language processing! It's amazing."

print(f"\nOriginal text: '{sample_text}'")
print("\nTokenization is required because:")
print("1. Neural networks work with numbers, not raw text")
print("2. We need to create a vocabulary mapping")
print("3. Different strategies affect model performance")

# ============================================================================
# PART 2: Simple Whitespace Tokenization
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Simple Whitespace Tokenization")
print("=" * 70)

# Basic split
tokens_simple = sample_text.split()
print(f"\nWhitespace split: {tokens_simple}")
print(f"Number of tokens: {len(tokens_simple)}")

print("\nProblem: 'processing!' includes punctuation")
print("Problem: 'It's' is kept as one token")

# ============================================================================
# PART 3: Better Tokenization with Regex
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Regex-Based Tokenization")
print("=" * 70)

def tokenize_regex(text):
    """Tokenize using regex to handle punctuation"""
    # Convert to lowercase
    text = text.lower()
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

tokens_regex = tokenize_regex(sample_text)
print(f"\nRegex tokenization: {tokens_regex}")
print(f"Number of tokens: {len(tokens_regex)}")
print("\n Better! Punctuation removed, lowercase")

# ============================================================================
# PART 4: NLTK Word Tokenization
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: NLTK Word Tokenization")
print("=" * 70)

try:
    import nltk
    nltk.download('punkt', quiet=True)
    
    tokens_nltk = nltk.word_tokenize(sample_text)
    print(f"\nNLTK tokenization: {tokens_nltk}")
    print(f"Number of tokens: {len(tokens_nltk)}")
    print("\n Notice: 'It's' split into 'It' + \"'s\"")
    print("Notice: Punctuation preserved as separate tokens")
    
    # Contractions example
    contraction_text = "I can't believe they're not going!"
    tokens_contractions = nltk.word_tokenize(contraction_text)
    print(f"\nContractions example: '{contraction_text}'")
    print(f"Tokens: {tokens_contractions}")
    
except ImportError:
    print("\nNLTK not installed. Install with: pip install nltk")

# ============================================================================
# PART 5: Character-Level Tokenization
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Character-Level Tokenization")
print("=" * 70)

def tokenize_characters(text):
    """Tokenize into individual characters"""
    return list(text)

text_short = "Hello NLP!"
tokens_char = tokenize_characters(text_short)

print(f"\nText: '{text_short}'")
print(f"Character tokens: {tokens_char}")
print(f"Number of tokens: {len(tokens_char)}")

print("\nAdvantages:")
print("+ Very small vocabulary (< 100 characters)")
print("+ Can handle any word, including typos")
print("+ No out-of-vocabulary problem")

print("\nDisadvantages:")
print("- Very long sequences (each word = many characters)")
print("- Individual characters carry little meaning")
print("- Computationally expensive for long texts")

# ============================================================================
# PART 6: Subword Tokenization (BPE Intuition)
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Subword Tokenization (BPE Intuition)")
print("=" * 70)

print("\nByte Pair Encoding (BPE) - Key idea:")
print("-" * 40)
print("""
1. Start with character-level vocabulary
2. Find most frequent pair of adjacent tokens
3. Merge that pair into new token
4. Repeat until desired vocabulary size
""")

# Simplified BPE demonstration
corpus = ["low", "low", "low", "lower", "lower", "newest", "newest", "newest", "newest"]

print(f"\nSample corpus: {corpus}")
print("\nBPE iterations (simplified):")

# Initial: character vocabulary
vocab = set()
for word in corpus:
    vocab.update(list(word))

print(f"\n1. Initial character vocab: {sorted(vocab)}")

# Simulate BPE merges
print("\n2. After merging 'e' + 's' -> 'es':")
print("   'newest' -> ['n', 'e', 'w', 'es', 't']")

print("\n3. After merging 'es' + 't' -> 'est':")
print("   'newest' -> ['n', 'e', 'w', 'est']")

print("\n4. After merging 'l' + 'o' -> 'lo':")
print("   'low' -> ['lo', 'w']")

print("\nResult: 'lowest' (unseen word) -> ['lo', 'w', 'est']")
print("Even unseen words can be represented through subwords!")

# ============================================================================
# PART 7: Using Keras Tokenizer
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Keras Tokenizer")
print("=" * 70)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample corpus
texts = [
    "I love natural language processing",
    "Deep learning is amazing",
    "NLP with neural networks is powerful",
    "I love deep learning models"
]

print("\nCorpus:")
for i, text in enumerate(texts, 1):
    print(f"  {i}. {text}")

# Create tokenizer
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

print(f"\nVocabulary (word -> index):")
for word, idx in list(tokenizer.word_index.items())[:15]:
    print(f"  '{word}': {idx}")

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)
print(f"\nText to sequences:")
for text, seq in zip(texts, sequences):
    print(f"  '{text}' -> {seq}")

# Pad sequences
padded = pad_sequences(sequences, maxlen=10, padding='post')
print(f"\nPadded sequences (maxlen=10, post-padding):")
print(padded)

# Test with new text containing OOV
new_text = ["I love transformers and GPT"]
new_seq = tokenizer.texts_to_sequences(new_text)
print(f"\nNew text with OOV words: '{new_text[0]}'")
print(f"Sequence: {new_seq[0]}")
print("Note: 'transformers', 'and', 'GPT' mapped to <OOV> (index 1)")

# ============================================================================
# PART 8: Hugging Face Tokenizers (Modern Approach)
# ============================================================================

print("\n" + "=" * 70)
print("PART 8: Hugging Face Tokenizers (Preview)")
print("=" * 70)

print("\nHugging Face provides production-ready tokenizers:")
print("-" * 50)

print("""
# Example with BERT tokenizer:
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "I love natural language processing!"

# Tokenize
tokens = tokenizer.tokenize(text)
# ['i', 'love', 'natural', 'language', 'processing', '!']

# Convert to IDs
input_ids = tokenizer.encode(text)
# [101, 1045, 2293, 3019, 2653, 6364, 999, 102]
# Note: 101 = [CLS], 102 = [SEP]

# Full encoding for model input
encoded = tokenizer(text, return_tensors='tf')
# Returns: {'input_ids': ..., 'attention_mask': ...}
""")

print("\nWhy use Hugging Face tokenizers?")
print("+ Pre-trained on massive corpora")
print("+ Consistent with pre-trained models")
print("+ Handles special tokens automatically")
print("+ Blazing fast (implemented in Rust)")

# ============================================================================
# PART 9: Building a Complete Tokenization Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("PART 9: Complete Tokenization Pipeline")
print("=" * 70)

class TextPreprocessor:
    """Complete text preprocessing pipeline"""
    
    def __init__(self, vocab_size=10000, max_length=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.fitted = False
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters (keep letters, numbers, spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit(self, texts):
        """Build vocabulary from corpus"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        self.tokenizer.fit_on_texts(cleaned_texts)
        self.fitted = True
        print(f"Vocabulary built: {len(self.tokenizer.word_index)} unique words")
    
    def transform(self, texts):
        """Convert texts to padded sequences"""
        if not self.fitted:
            raise ValueError("Must call fit() first!")
        
        cleaned_texts = [self.clean_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded
    
    def fit_transform(self, texts):
        """Fit and transform in one call"""
        self.fit(texts)
        return self.transform(texts)

# Demonstrate pipeline
corpus = [
    "I love NLP! It's fascinating.",
    "Deep learning models are amazing!!!",
    "Check out https://example.com for more info",
    "Natural language processing with Python"
]

print("\nOriginal corpus:")
for text in corpus:
    print(f"  {text}")

preprocessor = TextPreprocessor(vocab_size=100, max_length=10)
processed = preprocessor.fit_transform(corpus)

print(f"\nProcessed sequences (shape: {processed.shape}):")
print(processed)

# Test on new data
new_texts = ["I love Python programming"]
new_processed = preprocessor.transform(new_texts)
print(f"\nNew text: '{new_texts[0]}'")
print(f"Processed: {new_processed[0]}")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE: Tokenization Methods")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Tokenization converts text to processable units")
print("2. Word-level: intuitive but large vocabulary, OOV issues")
print("3. Character-level: no OOV but very long sequences")
print("4. Subword (BPE): best of both worlds, modern standard")
print("5. Always preprocess: lowercase, remove noise, normalize")
print("6. Keras Tokenizer handles vocabulary + numericalization")
print("7. Hugging Face for production with pre-trained models")

print("\n" + "=" * 70)

