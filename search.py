"""
Phase 3: Search CLI & Interactive Query Interface for IRE Project
Provides command-line, interactive, and benchmark demo search capabilities.
"""

import sys
import argparse
from ranker import BM25SearchEngine

def format_result(res: dict) -> str:
    out = []
    out.append(f"#{res['rank']} | Score: {res['score']} | [{res['flair']}] {res['title']}")
    if res['snippet']:
        out.append(f"   Snippet:  {res['snippet']}")
    out.append(f"   Metadata: Upvotes: {res['upvotes']} | Comments: {res['num_comments']} | Date: {res['date']}")
    out.append(f"   URL:      {res['url']}")
    return "\n".join(out)

def run_query(engine: BM25SearchEngine, query: str, top_k: int = 5, flair: str = None, min_score: int = None, expand: bool = True):
    print("\n" + "=" * 70)
    print(f"🔍 Query: '{query}'")
    if flair:
        print(f"   Filter Flair: {flair}")
    if min_score:
        print(f"   Filter Min Score: {min_score}")
        
    response = engine.search(
        query=query, 
        top_k=top_k, 
        filter_flair=flair, 
        min_score=min_score,
        expand=expand
    )
    
    print(f"⚡ Latency: {response['latency_ms']} ms | Total Matching Posts: {response['total_matches']:,}")
    print(f"🏷️  Stems:   {response['stems']}")
    if response.get("expanded_terms"):
        print(f"➕ Expanded: {response['expanded_terms']}")
    print("-" * 70)
    
    if not response['results']:
        print("❌ No matching documents found.")
    else:
        for item in response['results']:
            print(format_result(item))
            print()
    print("=" * 70)

def run_demo(engine: BM25SearchEngine):
    test_queries = [
        "physics wallah vs unacademy",
        "best books for organic chemistry",
        "mock test score stuck at 120",
        "percentile vs marks 2022",
        "allen test series review"
    ]
    print("\n🚀 Running Phase 3 Demo on 5 Standard IRE Test Queries...")
    for q in test_queries:
        run_query(engine, q, top_k=3)

def interactive_mode(engine: BM25SearchEngine):
    print("\n" + "=" * 70)
    print("🎓 IRE JEE/NEET Search Engine (Phase 3 Interactive CLI)")
    print("   Type your query and press Enter. Type 'exit' or 'q' to quit.")
    print("=" * 70)
    
    while True:
        try:
            query = input("\nSearch > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "q", "quit"]:
                print("Goodbye!")
                break
            run_query(engine, query, top_k=5)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting search engine.")
            break

def main():
    parser = argparse.ArgumentParser(description="IRE JEE/NEET BM25 Search Engine")
    parser.add_argument("query", nargs="*", help="Search query string")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--demo", action="store_true", help="Run benchmark demo queries")
    parser.add_argument("--flair", type=str, default=None, help="Filter by flair substring (e.g., 'discussion')")
    parser.add_argument("--min_score", type=int, default=None, help="Filter by minimum upvote score")
    parser.add_argument("--no-expand", action="store_true", help="Disable domain-dictionary query expansion")
    
    args = parser.parse_args()
    
    engine = BM25SearchEngine(index_dir="index")
    engine.load_index()
    
    if args.demo:
        run_demo(engine)
    elif args.query:
        query_str = " ".join(args.query)
        run_query(engine, query_str, top_k=args.top_k, flair=args.flair, min_score=args.min_score,
                  expand=not args.no_expand)
    else:
        interactive_mode(engine)

if __name__ == "__main__":
    main()
