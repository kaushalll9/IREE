"""
Phase 3: Inverted Index Builder for IRE Project
Builds an efficient field-aware inverted index from data/cleaned_corpus.jsonl
"""

import os
import re
import json
import gzip
import time
import pickle
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

# Standard English stopwords (offline-safe, no external downloads required)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", 
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", 
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", 
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", 
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", 
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", 
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", 
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

stemmer = PorterStemmer()

def tokenize(text: str, remove_stop: bool = True, stem: bool = True) -> list[str]:
    """
    Extracts alphanumeric tokens, converts to lowercase, filters stopwords,
    and applies Porter stemming.
    """
    if not text:
        return []
    
    # Tokenize words of length >= 2
    raw_tokens = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
    
    if remove_stop:
        tokens = [t for t in raw_tokens if t not in STOPWORDS]
    else:
        tokens = raw_tokens
        
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
        
    return tokens

def build_index(corpus_path: str = "data/cleaned_corpus.jsonl", index_dir: str = "index"):
    """
    Reads the cleaned JSONL corpus and builds:
    1. index/postings.pkl.gz  -> Term to posting list: term -> [(doc_id_int, tf_title, tf_body), ...]
    2. index/metadata.json.gz  -> Document metadata for display and length normalization
    3. index/stats.json        -> Global corpus statistics (total docs, avg lengths, vocab size)
    """
    print(f"[+] Starting Phase 3: Inverted Index Construction from '{corpus_path}'...")
    start_time = time.time()
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
    os.makedirs(index_dir, exist_ok=True)
    
    # Inverted Index structure:
    # term -> list of tuples (doc_int_id, tf_title, tf_body)
    postings = defaultdict(list)
    doc_metadata = []
    
    total_title_tokens = 0
    total_body_tokens = 0
    total_tokens = 0
    doc_count = 0
    
    print("[+] Tokenizing, stemming, and collecting postings...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for doc_int_id, line in enumerate(f):
            if not line.strip():
                continue
            
            record = json.loads(line)
            title = record.get("title", "")
            body = record.get("body", "")
            
            # Tokenize title and body separately
            title_tokens = tokenize(title)
            body_tokens = tokenize(body)
            
            len_title = len(title_tokens)
            len_body = len(body_tokens)
            len_total = len_title + len_body
            
            total_title_tokens += len_title
            total_body_tokens += len_body
            total_tokens += len_total
            doc_count += 1
            
            # Count term frequencies per field
            title_counts = Counter(title_tokens)
            body_counts = Counter(body_tokens)
            
            unique_terms = set(title_counts.keys()) | set(body_counts.keys())
            for term in unique_terms:
                tf_title = title_counts.get(term, 0)
                tf_body = body_counts.get(term, 0)
                postings[term].append((doc_int_id, tf_title, tf_body))
            
            # Create snippet for fast search result display
            snippet = body[:180] + ("..." if len(body) > 180 else "") if body else ""
            
            doc_metadata.append({
                "doc_int_id": doc_int_id,
                "doc_id": record.get("doc_id", ""),
                "title": title,
                "snippet": snippet,
                "author": record.get("author", ""),
                "score": int(record.get("score", 0)),
                "num_comments": int(record.get("num_comments", 0)),
                "flair": record.get("flair", "General"),
                "date": record.get("date", ""),
                "url": record.get("url", ""),
                "len_title": len_title,
                "len_body": len_body,
                "len_total": len_total
            })
            
            if (doc_count % 10000) == 0:
                print(f"    Indexed {doc_count:,} documents...")

    vocab_size = len(postings)
    avg_title_len = total_title_tokens / doc_count if doc_count else 0
    avg_body_len = total_body_tokens / doc_count if doc_count else 0
    avg_doc_len = total_tokens / doc_count if doc_count else 0
    
    elapsed_build = time.time() - start_time
    print(f"\n[✓] Index built in memory ({elapsed_build:.2f}s).")
    print(f"    Documents Indexed: {doc_count:,}")
    print(f"    Vocabulary Size:   {vocab_size:,} unique terms")
    print(f"    Average Doc Length: {avg_doc_len:.1f} tokens (Title: {avg_title_len:.1f}, Body: {avg_body_len:.1f})")

    # Serialize postings to gzip-compressed pickle
    postings_file = os.path.join(index_dir, "postings.pkl.gz")
    print(f"[+] Compressing & saving postings to '{postings_file}'...")
    with gzip.open(postings_file, "wb", compresslevel=4) as f:
        pickle.dump(dict(postings), f, protocol=pickle.HIGHEST_PROTOCOL)

    # Serialize metadata to gzip-compressed json
    meta_file = os.path.join(index_dir, "metadata.json.gz")
    print(f"[+] Saving document metadata to '{meta_file}'...")
    with gzip.open(meta_file, "wt", encoding="utf-8") as f:
        json.dump(doc_metadata, f, ensure_ascii=False)

    # Save stats summary
    stats = {
        "total_docs": doc_count,
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "avg_doc_len": round(avg_doc_len, 2),
        "avg_title_len": round(avg_title_len, 2),
        "avg_body_len": round(avg_body_len, 2),
        "build_time_seconds": round(elapsed_build, 2),
        "postings_file": postings_file,
        "metadata_file": meta_file
    }
    stats_file = os.path.join(index_dir, "stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    total_time = time.time() - start_time
    postings_size_mb = os.path.getsize(postings_file) / (1024 * 1024)
    meta_size_mb = os.path.getsize(meta_file) / (1024 * 1024)
    print(f"\n[✓] Phase 3 Indexing complete in {total_time:.2f}s!")
    print(f"    - Postings size: {postings_size_mb:.2f} MB")
    print(f"    - Metadata size: {meta_size_mb:.2f} MB")
    print(f"    - Stats saved:   {stats_file}")
    
    return stats

if __name__ == "__main__":
    build_index()
