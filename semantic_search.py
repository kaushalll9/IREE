"""
Semantic retrieval with a multilingual Sentence Transformer.

Document embeddings are computed once from data/cleaned_corpus.jsonl (title + body),
L2-normalised and cached in index/. Row i of the matrix is BM25 metadata[i], so results
reuse the BM25 engine's metadata untouched. Rebuild with: python semantic_search.py
"""

import os
import json
import time
import hashlib
from typing import Any, Dict, Optional

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CORPUS_PATH = "data/cleaned_corpus.jsonl"
BODY_CHARS = 1000  # model truncates at 128 word-pieces anyway; this just keeps encoding fast


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SemanticSearchEngine:
    def __init__(self, bm25_engine, index_dir: str = "index",
                 corpus_path: str = CORPUS_PATH, model_name: str = MODEL_NAME):
        # Shares the BM25 engine's metadata so doc ids, titles, URLs are identical across modes.
        self.bm25 = bm25_engine
        self.index_dir = index_dir
        self.corpus_path = corpus_path
        self.model_name = model_name
        self.emb_path = os.path.join(index_dir, "doc_embeddings.npy")
        self.info_path = os.path.join(index_dir, "doc_embeddings.json")
        self.model = None
        self.embeddings = None

    def load(self):
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed

        if self.bm25.metadata is None:
            self.bm25.load_index()
        self.model = SentenceTransformer(self.model_name)  # auto-selects CUDA / Apple MPS / CPU
        print(f"[+] Semantic model loaded on device: {self.model.device}")

        expected = {"model": self.model_name,
                    "corpus_sha1": _file_sha1(self.corpus_path),
                    "num_docs": len(self.bm25.metadata)}
        if self._cache_matches(expected):
            self.embeddings = np.load(self.emb_path)
            print(f"[✓] Loaded cached embeddings {self.embeddings.shape} from '{self.emb_path}'.")
        else:
            print("[!] Embedding cache missing or stale (corpus/model changed); rebuilding...")
            self._build(expected)

    def _cache_matches(self, expected: dict) -> bool:
        if not (os.path.exists(self.emb_path) and os.path.exists(self.info_path)):
            return False
        with open(self.info_path, encoding="utf-8") as f:
            return json.load(f) == expected

    def _build(self, info: dict, batch_size: int = 64):
        texts = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:  # same blank-line skipping as indexer.py, so row i == metadata[i]
                if not line.strip():
                    continue
                rec = json.loads(line)
                meta = self.bm25.metadata[len(texts)]
                if rec.get("doc_id", "") != meta["doc_id"]:
                    raise ValueError(f"Corpus/index mismatch at row {len(texts)}: re-run indexer.py")
                texts.append(f"{rec.get('title', '')}. {(rec.get('body') or '')[:BODY_CHARS]}")
        if len(texts) != len(self.bm25.metadata):
            raise ValueError(f"Corpus has {len(texts)} docs but index has {len(self.bm25.metadata)}")

        t0 = time.time()
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True)
        self.embeddings = emb.astype(np.float32)
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(self.emb_path, self.embeddings)
        with open(self.info_path, "w", encoding="utf-8") as f:
            json.dump(info, f)
        print(f"[✓] Encoded {len(texts):,} docs in {time.time() - t0:.1f}s -> '{self.emb_path}'.")

    def search(self, query: str, top_k: int = 10, filter_flair: Optional[str] = None,
               min_score: Optional[int] = None) -> Dict[str, Any]:
        """Cosine-similarity top-k. Same response shape as BM25SearchEngine.search."""
        if self.embeddings is None:
            self.load()
        start = time.time()
        meta = self.bm25.metadata

        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.embeddings @ q  # normalised vectors: dot product == cosine similarity

        if filter_flair or min_score is not None:
            # ponytail: O(N) Python mask per filtered query (~10ms); precompute arrays if it shows up in profiles
            keep = np.fromiter(
                ((not filter_flair or filter_flair.lower() in m.get("flair", "").lower())
                 and (min_score is None or m.get("score", 0) >= min_score) for m in meta),
                dtype=bool, count=len(meta))
            sims = np.where(keep, sims, -np.inf)
            total = int(keep.sum())
        else:
            total = len(meta)

        k = min(top_k, total)
        top = np.argpartition(-sims, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.argsort(-sims[top])]

        results = []
        for rank, i in enumerate(top, start=1):
            m = meta[i]
            results.append({
                "rank": rank,
                "doc_id": m["doc_id"],
                "score": round(float(sims[i]), 4),
                "title": m["title"],
                "snippet": m["snippet"],
                "author": m["author"],
                "flair": m["flair"],
                "upvotes": m["score"],
                "num_comments": m["num_comments"],
                "date": m["date"],
                "url": m["url"],
                "method": "semantic",
            })
        return {
            "query": query,
            "stems": [],
            "expanded_terms": {},
            "total_matches": total,
            "latency_ms": round((time.time() - start) * 1000, 2),
            "results": results,
        }


if __name__ == "__main__":
    from ranker import BM25SearchEngine
    engine = BM25SearchEngine(index_dir="index")
    engine.load_index()
    sem = SemanticSearchEngine(engine)
    sem.load()  # builds/refreshes the cache if needed
    for r in sem.search("JEE ki preparation kaise karu?", top_k=5)["results"]:
        print(f"#{r['rank']} {r['score']:.3f} {r['title']}")
