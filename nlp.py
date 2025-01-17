# nlp.py

from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline
from sklearn.cluster import KMeans
import json
import os
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/nlp_{datetime.now().strftime("%Y%m%d")}.log'
)
logger = logging.getLogger(__name__)

# Initialize models with error handling
try:
    # Summarization model with improved max length
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

    # Sentence Transformer model for better embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("NLP models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing NLP models: {e}")
    raise

# Enhanced stopwords list
STOPWORDS = {
    # Common English stopwords
    'the', 'and', 'a', 'an', 'is', 'of', 'to', 'in', 'for', 'on', 'at', 'by',
    'with', 'from', 'this', 'that', 'it', 'as', 'are', 'was', 'were', 'be',
    'has', 'have', 'had', 'not', 'or', 'if', 'we', 'you', 'i', 'our', 'your',

    # Additional common words
    'about', 'above', 'after', 'again', 'all', 'am', 'any', 'because',
    'been', 'before', 'being', 'below', 'between', 'both', 'but', 'can',
    'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself',

    # Common technical terms that shouldn't define categories
    'file', 'document', 'page', 'section', 'chapter', 'paragraph'
}


def summarize_text(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """
    Generate a concise summary of the input text with improved handling of edge cases.

    Args:
        text: Input text to summarize
        max_length: Maximum length of the summary
        min_length: Minimum length of the summary

    Returns:
        str: Summarized text
    """
    try:
        # Input validation
        if not text or not text.strip():
            return "No text provided for summarization."

        # Clean and prepare text
        text = clean_text(text)

        # Handle text that's shorter than min_length
        if len(text.split()) < min_length:
            return text

        # Generate summary with error handling
        summary = summarizer(text,
                             max_length=max_length,
                             min_length=min_length,
                             do_sample=False)

        return summary[0]['summary_text']

    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return "Error generating summary. Please try again."


def generate_embeddings(text_list: List[str]) -> torch.Tensor:
    """
    Generate embeddings for a list of texts with improved batching and error handling.

    Args:
        text_list: List of texts to generate embeddings for

    Returns:
        torch.Tensor: Generated embeddings
    """
    try:
        if not text_list:
            return torch.tensor([])

        # Clean texts
        cleaned_texts = [clean_text(text) for text in text_list]

        # Generate embeddings with batching
        embeddings = embedding_model.encode(cleaned_texts,
                                            convert_to_tensor=True,
                                            batch_size=32,
                                            show_progress_bar=False)

        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return torch.tensor([])


def clean_text(text: str) -> str:
    """
    Clean and normalize text for better processing.

    Args:
        text: Input text to clean

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove special characters but keep essential punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?()-]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def load_user_preferences(pref_path: str = 'user_preferences.json') -> Dict:
    """
    Load user preferences with validation and defaults.

    Args:
        pref_path: Path to the preferences file

    Returns:
        Dict: User preferences
    """
    try:
        if os.path.exists(pref_path):
            with open(pref_path, 'r', encoding='utf-8') as f:
                prefs = json.load(f)

            # Validate preferences structure
            if not isinstance(prefs, dict):
                logger.warning("Invalid preferences format, using defaults")
                return {}

            return prefs
    except Exception as e:
        logger.error(f"Error loading preferences: {e}")

    return {}


def save_user_preferences(preferences: Dict, pref_path: str = 'user_preferences.json'):
    """
    Save user preferences with backup and validation.

    Args:
        preferences: Preferences to save
        pref_path: Path to save preferences
    """
    try:
        # Create backup of existing preferences
        if os.path.exists(pref_path):
            backup_path = f"{pref_path}.backup"
            os.replace(pref_path, backup_path)

        # Save new preferences
        with open(pref_path, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, ensure_ascii=False, indent=4)

    except Exception as e:
        logger.error(f"Error saving preferences: {e}")

        # Restore backup if available
        if os.path.exists(f"{pref_path}.backup"):
            os.replace(f"{pref_path}.backup", pref_path)


def generate_category_name(docs: List[Dict]) -> str:
    """
    Generate a meaningful category name based on document content with improved analysis.

    Args:
        docs: List of documents to analyze

    Returns:
        str: Generated category name
    """
    try:
        # Extract and clean text from all documents
        all_text = " ".join(doc['content'] for doc in docs if doc['content'])
        all_text = clean_text(all_text.lower())

        # Tokenize and count words
        words = re.findall(r'\w+', all_text)
        word_freq = Counter(word for word in words
                            if word not in STOPWORDS and len(word) > 2)

        if not word_freq:
            return "Miscellaneous"

        # Get most frequent meaningful words
        top_words = [word.title() for word, _ in word_freq.most_common(2)]

        if not top_words:
            return "Miscellaneous"

        return " ".join(top_words)

    except Exception as e:
        logger.error(f"Error generating category name: {e}")
        return "Category"


def cluster_documents(documents: List[Dict],
                      embeddings: torch.Tensor,
                      n_clusters: int = 3) -> Dict[int, Dict]:
    """
    Cluster documents with improved algorithm and error handling.

    Args:
        documents: List of documents to cluster
        embeddings: Document embeddings
        n_clusters: Number of clusters to create

    Returns:
        Dict: Clustered documents
    """
    try:
        if len(documents) == 0:
            return {}

        if len(documents) < n_clusters:
            n_clusters = len(documents)

        # Convert embeddings to numpy for clustering
        emb_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings

        # Perform clustering with multiple attempts
        kmeans = KMeans(n_clusters=n_clusters,
                        n_init=10,
                        random_state=42)

        labels = kmeans.fit_predict(emb_np)

        # Organize documents into categories
        categories = {i: {"name": "Placeholder", "files": []} for i in range(n_clusters)}

        for doc, label in zip(documents, labels):
            categories[label]["files"].append(doc)

        # Generate meaningful names for categories
        for cat_id, cat_info in categories.items():
            cat_name = generate_category_name(cat_info["files"])
            categories[cat_id]["name"] = cat_name if cat_name.strip() else f"Category {cat_id + 1}"

        return categories

    except Exception as e:
        logger.error(f"Error clustering documents: {e}")
        return {}


def semantic_search(query: str,
                    documents: List[Dict],
                    embeddings: torch.Tensor,
                    top_k: int = 5) -> List[Tuple[Dict, float]]:
    """
    Perform semantic search with improved relevance scoring.

    Args:
        query: Search query
        documents: Documents to search
        embeddings: Document embeddings
        top_k: Number of results to return

    Returns:
        List[Tuple]: List of (document, score) pairs
    """
    try:
        # Input validation
        if not query or not documents or embeddings is None:
            return []

        # Generate query embedding
        query_embedding = embedding_model.encode(clean_text(query),
                                                 convert_to_tensor=True)

        # Calculate similarity scores
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

        # Get top results
        top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))

        # Prepare results with score normalization
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            # Normalize score to 0-1 range
            normalized_score = (score.item() + 1) / 2
            if normalized_score > 0.3:  # Threshold for relevance
                results.append((documents[idx], normalized_score))

        return results

    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []


def incorporate_user_feedback(categories: Dict[int, Dict],
                              user_adjustments: Dict[str, Any]):
    """
    Incorporate user feedback for improved categorization.

    Args:
        categories: Current document categories
        user_adjustments: User's adjustments to categories
    """
    try:
        user_prefs = load_user_preferences()

        if "history" not in user_prefs:
            user_prefs["history"] = []

        # Add current categorization to history
        user_prefs["history"].append({
            "timestamp": datetime.now().isoformat(),
            "categories": categories,
            "adjustments": user_adjustments
        })

        # Keep only last 10 history entries
        user_prefs["history"] = user_prefs["history"][-10:]

        save_user_preferences(user_prefs)

    except Exception as e:
        logger.error(f"Error incorporating user feedback: {e}")