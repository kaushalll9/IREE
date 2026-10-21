"""
Phase 3: BM25 / BM25F Ranking Engine for IRE Project
Implements Okapi BM25 and Field-Aware BM25 (BM25F) scoring with fast retrieval.
"""

import os
import math
import gzip
import json
import pickle
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional
from indexer import tokenize

class BM25SearchEngine:
    def __init__(self, index_dir: str = "index"):
        self.index_dir = index_dir
        self.postings = None
        self.metadata = None
        self.stats = None
        self.total_docs = 0
        self.avg_title_len = 0.0
        self.avg_body_len = 0.0
        self.avg_doc_len = 0.0
        self.idf_cache = {}
        
    def load_index(self):
        """Loads inverted index, metadata, and stats into memory."""
        stats_path = os.path.join(self.index_dir, "stats.json")
        postings_path = os.path.join(self.index_dir, "postings.pkl.gz")
        meta_path = os.path.join(self.index_dir, "metadata.json.gz")
        
        if not os.path.exists(stats_path) or not os.path.exists(postings_path):
            raise FileNotFoundError(
                f"Index files not found in '{self.index_dir}'. Please run 'python3 indexer.py' first!"
            )
            
        t0 = time.time()
        print(f"[+] Loading index from '{self.index_dir}'...")
        
        with open(stats_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)
            self.total_docs = self.stats["total_docs"]
            self.avg_title_len = self.stats.get("avg_title_len", 1.0) or 1.0
            self.avg_body_len = self.stats.get("avg_body_len", 1.0) or 1.0
            self.avg_doc_len = self.stats.get("avg_doc_len", 1.0) or 1.0
            
        with gzip.open(postings_path, "rb") as f:
            self.postings = pickle.load(f)
            
        with gzip.open(meta_path, "rt", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.idf_cache.clear()
        print(f"[✓] Index loaded successfully in {time.time() - t0:.2f}s ({self.total_docs:,} docs, {len(self.postings):,} terms).")

    def get_idf(self, term: str) -> float:
        """
        Calculates Robertson / Lucene standard BM25 IDF:
        IDF = ln(1 + (N - df + 0.5) / (df + 0.5))
        Guarantees non-negative scores even for frequent terms.
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
            
        postings_list = self.postings.get(term)
        if not postings_list:
            idf = 0.0
        else:
            df = len(postings_list)
            # Standard smoothed BM25 IDF
            idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
            
        self.idf_cache[term] = idf
        return idf

    def search(
        self,
        query: str,
        top_k: int = 10,
        k1: float = 1.2,
        b_title: float = 0.75,
        b_body: float = 0.75,
        w_title: float = 2.5,
        w_body: float = 1.0,
        use_engagement_boost: bool = True,
        filter_flair: Optional[str] = None,
        min_score: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes a BM25F search across Title and Body fields.
        Returns top_k ranked documents with scores and execution stats.
        """
        if self.postings is None:
            self.load_index()
            
        start_time = time.time()
        
        # 1. Tokenize & stem query
        query_stems = tokenize(query)
        if not query_stems:
            return {
                "query": query,
                "stems": [],
                "total_matches": 0,
                "latency_ms": 0.0,
                "results": []
            }
            
        # 2. Accumulate BM25 scores for candidate documents
        doc_scores = {}
        matched_terms = defaultdict(set)
        
        for stem in query_stems:
            postings_list = self.postings.get(stem)
            if not postings_list:
                continue
                
            idf = self.get_idf(stem)
            if idf <= 0:
                continue
                
            for doc_id, tf_title, tf_body in postings_list:
                # Fast metadata check
                meta = self.metadata[doc_id]
                
                # Length normalization per field
                norm_tf_title = tf_title / (1.0 - b_title + b_title * (meta["len_title"] / self.avg_title_len))
                norm_tf_body = tf_body / (1.0 - b_body + b_body * (meta["len_body"] / self.avg_body_len))
                
                # Combined weighted term frequency
                weighted_tf = (w_title * norm_tf_title) + (w_body * norm_tf_body)
                
                # Saturation component
                term_score = idf * ((weighted_tf * (k1 + 1.0)) / (weighted_tf + k1))
                
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score
                matched_terms[doc_id].add(stem)

        if not doc_scores:
            return {
                "query": query,
                "stems": query_stems,
                "total_matches": 0,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "results": []
            }

        # 3. Apply post-scoring filtering & optional engagement boost
        candidates = []
        for doc_id, base_score in doc_scores.items():
            meta = self.metadata[doc_id]
            
            # Apply filters if specified
            if filter_flair and filter_flair.lower() not in meta.get("flair", "").lower():
                continue
            if min_score is not None and meta.get("score", 0) < min_score:
                continue
                
            final_score = base_score
            
            # Engagement boost: logarithmic bonus for highly upvoted / engaged community posts
            if use_engagement_boost:
                reddit_score = max(0, meta.get("score", 0))
                engagement_multiplier = 1.0 + 0.05 * math.log10(1.0 + reddit_score)
                final_score *= engagement_multiplier
                
            candidates.append((doc_id, final_score, len(matched_terms[doc_id])))

        # 4. Sort by score descending (with term coverage as tiebreaker)
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_candidates = candidates[:top_k]
        
        # 5. Build final result objects
        results = []
        for rank, (doc_id, score, term_cov) in enumerate(top_candidates, start=1):
            meta = self.metadata[doc_id]
            results.append({
                "rank": rank,
                "doc_id": meta["doc_id"],
                "score": round(score, 4),
                "title": meta["title"],
                "snippet": meta["snippet"],
                "author": meta["author"],
                "flair": meta["flair"],
                "upvotes": meta["score"],
                "num_comments": meta["num_comments"],
                "date": meta["date"],
                "url": meta["url"],
                "matched_terms": list(matched_terms[doc_id])
            })

        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "query": query,
            "stems": query_stems,
            "total_matches": len(candidates),
            "latency_ms": latency_ms,
            "results": results
        }
