import re
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import gc
import spaces

# Global variable to cache the model
_cached_model = None

def get_embedding_model():
    """Load and cache the embedding model to avoid reloading"""
    global _cached_model
    if _cached_model is None:
        print("Loading GTE-large model for the first time...")
        _cached_model = SentenceTransformer("thenlper/gte-large")
        # Check if CUDA is available and move to GPU
        if torch.cuda.is_available():
            print("CUDA detected - using GPU for embeddings")
            _cached_model = _cached_model.cuda()
        else:
            print("CUDA not available - using CPU for embeddings")
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

@spaces.GPU(duration=120)  # 2 minute GPU limit for shared demo
def compute_embeddings_gpu(texts, model_name="thenlper/gte-large", batch_size=32):
    """Compute embeddings on GPU with batching for efficient processing"""
    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_tensor=False  # Return numpy for memory efficiency
    )
    
    return embeddings

def run_embed_match_batched(input_list, target_list, progress_callback=None):
    """
    Run semantic embedding matching with batched processing for large datasets.
    Processes in chunks to avoid GPU timeout and memory issues.
    """
    # Use minimal cleaning for embeddings
    input_list_clean = clean_text_for_embedding(input_list)
    target_list_clean = clean_text_for_embedding(target_list)
    
    total_inputs = len(input_list_clean)
    total_targets = len(target_list_clean)
    
    print(f"Processing {total_inputs} inputs against {total_targets} targets")
    
    # For very large datasets, process in batches
    MAX_BATCH_SIZE = 30000  # Process max 30k at a time for faster processing
    EMBEDDING_BATCH_SIZE = 64 if torch.cuda.is_available() else 32
    
    # First, encode all targets (usually smaller set)
    if progress_callback:
        progress_callback(0.1, desc="Encoding target descriptions...")
    
    if total_targets > MAX_BATCH_SIZE:
        # Process targets in chunks if too large
        target_vecs = []
        for i in range(0, total_targets, MAX_BATCH_SIZE):
            batch = target_list_clean[i:i+MAX_BATCH_SIZE]
            batch_vecs = compute_embeddings_gpu(batch, batch_size=EMBEDDING_BATCH_SIZE)
            target_vecs.append(batch_vecs)
            if progress_callback:
                progress = 0.1 + (0.3 * (i + len(batch)) / total_targets)
                progress_callback(progress, desc=f"Encoding targets: {i+len(batch)}/{total_targets}")
        target_vecs = np.vstack(target_vecs)
    else:
        target_vecs = compute_embeddings_gpu(target_list_clean, batch_size=EMBEDDING_BATCH_SIZE)
    
    # Process inputs in batches
    matches = []
    scores = []
    
    for batch_start in range(0, total_inputs, MAX_BATCH_SIZE):
        batch_end = min(batch_start + MAX_BATCH_SIZE, total_inputs)
        input_batch = input_list_clean[batch_start:batch_end]
        
        if progress_callback:
            progress = 0.4 + (0.4 * batch_start / total_inputs)
            progress_callback(progress, desc=f"Processing batch {batch_start}-{batch_end} of {total_inputs}")
        
        # Encode input batch
        input_batch_vecs = compute_embeddings_gpu(input_batch, batch_size=EMBEDDING_BATCH_SIZE)
        
        # Calculate similarities for this batch
        similarity_matrix = cosine_similarity(input_batch_vecs, target_vecs)
        
        # Find best matches for this batch
        for row in similarity_matrix:
            best_idx = np.argmax(row)
            best_score = row[best_idx]
            best_match = target_list[best_idx]
            
            matches.append(best_match)
            scores.append(float(best_score))
        
        # Clean up memory
        del input_batch_vecs
        del similarity_matrix
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if progress_callback:
            progress = 0.4 + (0.4 * batch_end / total_inputs)
            progress_callback(progress, desc=f"Processed {batch_end}/{total_inputs} inputs")
    
    if progress_callback:
        progress_callback(0.9, desc="Finalizing results...")
    
    return {"match": matches, "score": scores}

def run_embed_match(input_list, target_list):
    """
    Original function for backward compatibility.
    Automatically uses batched processing for large datasets.
    """
    total_items = len(input_list) + len(target_list)
    
    # Use batched processing for large datasets
    if total_items > 20000:  # If combined size > 20k, use batching
        print(f"Large dataset detected ({total_items} items), using batched processing...")
        return run_embed_match_batched(input_list, target_list)
    
    # Original implementation for smaller datasets
    input_list_clean = clean_text_for_embedding(input_list)
    target_list_clean = clean_text_for_embedding(target_list)
    
    # Load the model (cached)
    model = get_embedding_model()
    
    # Generate embeddings with normalization for cosine similarity
    import torch
    batch_size = 64 if torch.cuda.is_available() else 32
    
    print(f"Encoding {len(input_list_clean)} input descriptions...")
    input_vecs = model.encode(input_list_clean, 
                             normalize_embeddings=True, 
                             show_progress_bar=False,
                             batch_size=batch_size,
                             convert_to_tensor=True)
    
    print(f"Encoding {len(target_list_clean)} target descriptions...")
    target_vecs = model.encode(target_list_clean, 
                              normalize_embeddings=True, 
                              show_progress_bar=False,
                              batch_size=batch_size,
                              convert_to_tensor=True)
    
    # Convert to numpy for cosine similarity calculation
    if torch.is_tensor(input_vecs):
        input_vecs = input_vecs.cpu().numpy()
    if torch.is_tensor(target_vecs):
        target_vecs = target_vecs.cpu().numpy()
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(input_vecs, target_vecs)
    
    matches = []
    scores = []
    
    for i, row in enumerate(similarity_matrix):
        best_idx = np.argmax(row)
        best_score = row[best_idx]
        best_match = target_list[best_idx]
        
        matches.append(best_match)
        scores.append(float(best_score))
    
    return {"match": matches, "score": scores}