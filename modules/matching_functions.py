import re
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Global variable to cache the model
_cached_model = None

def get_embedding_model():
    """Load and cache the embedding model to avoid reloading"""
    global _cached_model
    if _cached_model is None:
        print("Loading GTE-large model for the first time...")
        _cached_model = SentenceTransformer("thenlper/gte-large")
    return _cached_model

def clean_text_simple(text_list):
    """
    Clean text by removing punctuation and extra spaces.
    Note: For embedding models, minimal cleaning is often better as they can
    handle punctuation and capitalization to understand context better.
    """
    cleaned = []
    for text in text_list:
        # Convert to string and strip whitespace
        text = str(text).strip()
        # Remove only excessive punctuation and normalize spaces
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'[^\w\s,.-]', '', text)  # Keep letters, numbers, spaces, commas, periods, hyphens
        cleaned.append(text.lower())
    return cleaned

def clean_text_for_embedding(text_list):
    """
    Minimal cleaning for embedding models - they work better with original text.
    Only removes excessive whitespace and ensures string format.
    """
    cleaned = []
    for text in text_list:
        # Just ensure it's a string and normalize whitespace
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        cleaned.append(text)
    return cleaned

def run_fuzzy_match(input_list, target_list, clean=True):
    """Run fuzzy string matching"""
    if clean:
        input_list = clean_text_simple(input_list)
        target_list = clean_text_simple(target_list)
    
    matches = []
    scores = []
    
    for input_desc in input_list:
        best_match, score, _ = process.extractOne(
            input_desc, 
            target_list, 
            scorer=fuzz.ratio
        )
        matches.append(best_match)
        scores.append(score)
    
    return {"match": matches, "score": scores}

def run_tfidf_match(input_list, target_list, clean=True):
    """Run TF-IDF matching with cosine similarity"""
    if clean:
        input_list = clean_text_simple(input_list)
        target_list = clean_text_simple(target_list)
    
    # Combine for consistent vocabulary
    combined = input_list + target_list
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectorizer.fit(combined)
    
    tfidf_input = vectorizer.transform(input_list)
    tfidf_target = vectorizer.transform(target_list)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_input, tfidf_target)
    
    matches = []
    scores = []
    
    for i, row in enumerate(similarity_matrix):
        best_idx = np.argmax(row)
        best_score = row[best_idx]
        best_match = target_list[best_idx]
        
        matches.append(best_match)
        scores.append(float(best_score))
    
    return {"match": matches, "score": scores}

def run_embed_match(input_list, target_list):
    """
    Run semantic embedding matching using GTE-large model.
    This model works best with original text (minimal cleaning).
    """
    # Use minimal cleaning for embeddings - preserve original text structure
    input_list_clean = clean_text_for_embedding(input_list)
    target_list_clean = clean_text_for_embedding(target_list)
    
    # Load the model (cached)
    model = get_embedding_model()
    
    # Generate embeddings with normalization for cosine similarity
    # The model understands context better with original capitalization and punctuation
    print(f"Encoding {len(input_list_clean)} input descriptions...")
    input_vecs = model.encode(input_list_clean, 
                             normalize_embeddings=True, 
                             show_progress_bar=False,
                             batch_size=32)
    
    print(f"Encoding {len(target_list_clean)} target descriptions...")
    target_vecs = model.encode(target_list_clean, 
                              normalize_embeddings=True, 
                              show_progress_bar=False,
                              batch_size=32)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(input_vecs, target_vecs)
    
    matches = []
    scores = []
    
    for i, row in enumerate(similarity_matrix):
        best_idx = np.argmax(row)
        best_score = row[best_idx]
        best_match = target_list[best_idx]  # Return original text, not cleaned
        
        matches.append(best_match)
        scores.append(float(best_score))
    
    # Print some debug info about score distribution
    scores_array = np.array(scores)
    print(f"Score statistics - Min: {scores_array.min():.3f}, Max: {scores_array.max():.3f}, Mean: {scores_array.mean():.3f}, Median: {np.median(scores_array):.3f}")
    
    return {"match": matches, "score": scores}