import os
import re
import html
import json
import time
from datetime import datetime, timezone
from collections import Counter

def clean_text(text: str) -> str:
    """Cleans Reddit markdown, HTML entities, and formatting artifacts."""
    if not text:
        return ""
    
    # 1. Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(text)
    
    # 2. Convert markdown links [anchor text](http...) -> anchor text
    text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', text)
    
    # 3. Strip standalone URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # 4. Remove Reddit spoiler / formatting markers (>!, !<, **, ~~, etc.)
    text = re.sub(r'>!|!<', '', text)
    text = re.sub(r'[\*_~`#]', ' ', text)
    
    # 5. Normalize whitespace and newlines
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()

def preprocess_corpus(input_file="data/JEENEETards_posts.jsonl", output_file="data/cleaned_corpus.jsonl"):
    print(f"[+] Starting Phase 2: Data Cleaning & Preprocessing on '{input_file}'...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found!")

    stats = {
        "total_raw": 0,
        "dropped_deleted_title": 0,
        "dropped_automod": 0,
        "empty_body_converted": 0,
        "cleaned_records": 0,
        "flair_counts": Counter(),
        "total_tokens": 0,
        "min_timestamp": float("inf"),
        "max_timestamp": 0
    }

    start_time = time.time()
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            stats["total_raw"] += 1
            record = json.loads(line)
            
            # 1. Title validation
            raw_title = record.get("title") or ""
            title_stripped = raw_title.strip()
            
            # Check for deleted/removed titles
            if not title_stripped or title_stripped.lower() in ["[deleted]", "[removed]", "[deleted by user]"]:
                stats["dropped_deleted_title"] += 1
                continue
            
            # 2. Check author (drop AutoModerator bots)
            author = record.get("author") or ""
            if author.lower() == "automoderator":
                stats["dropped_automod"] += 1
                continue
            
            # 3. Clean selftext (body)
            raw_body = record.get("selftext") or ""
            raw_body_stripped = raw_body.strip()
            if raw_body_stripped.lower() in ["[deleted]", "[removed]"]:
                clean_body = ""
                stats["empty_body_converted"] += 1
            else:
                clean_body = clean_text(raw_body)
                if not clean_body:
                    stats["empty_body_converted"] += 1

            clean_title = clean_text(raw_title)
            if not clean_title:
                stats["dropped_deleted_title"] += 1
                continue

            # 4. Construct unified searchable full_text
            full_text = f"{clean_title}. {clean_body}".strip() if clean_body else clean_title

            # 5. Extract metadata
            created_utc = int(record.get("created_utc") or 0)
            if created_utc > 0:
                stats["min_timestamp"] = min(stats["min_timestamp"], created_utc)
                stats["max_timestamp"] = max(stats["max_timestamp"], created_utc)
                date_str = datetime.fromtimestamp(created_utc, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                date_str = ""

            flair = record.get("link_flair_text")
            flair = flair.strip() if flair else "General"
            stats["flair_counts"][flair] += 1

            tokens = full_text.split()
            stats["total_tokens"] += len(tokens)

            # 6. Standardized Document Schema
            clean_doc = {
                "doc_id": record.get("id"),
                "title": clean_title,
                "body": clean_body,
                "full_text": full_text,
                "author": author if author not in ["[deleted]", "[removed]"] else "[anonymous]",
                "score": int(record.get("score") or 0),
                "num_comments": int(record.get("num_comments") or 0),
                "flair": flair,
                "created_utc": created_utc,
                "date": date_str,
                "url": record.get("url") or "",
                "permalink": f"https://www.reddit.com{record.get('permalink')}" if record.get("permalink") else "",
                "token_count": len(tokens)
            }

            fout.write(json.dumps(clean_doc, ensure_ascii=False) + "\n")
            stats["cleaned_records"] += 1

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    avg_tokens = stats["total_tokens"] / max(stats["cleaned_records"], 1)
    
    first_date = datetime.fromtimestamp(stats["min_timestamp"], timezone.utc).strftime("%Y-%m-%d") if stats["min_timestamp"] != float("inf") else "N/A"
    last_date = datetime.fromtimestamp(stats["max_timestamp"], timezone.utc).strftime("%Y-%m-%d") if stats["max_timestamp"] > 0 else "N/A"

    print("\n" + "=" * 50)
    print("      PHASE 2 DATA CLEANING COMPLETE")
    print("=" * 50)
    print(f"Total raw records processed:       {stats['total_raw']:,}")
    print(f"Dropped (deleted/removed titles):   {stats['dropped_deleted_title']:,}")
    print(f"Dropped (AutoModerator posts):      {stats['dropped_automod']:,}")
    print(f"Selftexts cleaned/empty converted:  {stats['empty_body_converted']:,}")
    print(f"Cleaned corpus document count:     {stats['cleaned_records']:,}")
    print(f"Average tokens per document:        {avg_tokens:.1f}")
    print(f"Corpus date span:                  {first_date} to {last_date}")
    print(f"Cleaned corpus size:               {file_size_mb:.2f} MB")
    print(f"Output file:                       {output_file}")
    print(f"Time taken:                        {elapsed:.2f}s")
    print("=" * 50)

    # Save summary report metadata
    summary = {
        "total_raw": stats["total_raw"],
        "cleaned_records": stats["cleaned_records"],
        "dropped_deleted_title": stats["dropped_deleted_title"],
        "dropped_automod": stats["dropped_automod"],
        "empty_body_converted": stats["empty_body_converted"],
        "cleaned_size_mb": round(file_size_mb, 2),
        "avg_tokens": round(avg_tokens, 1),
        "first_date": first_date,
        "last_date": last_date,
        "top_flairs": stats["flair_counts"].most_common(10)
    }
    with open("data/corpus_summary.json", "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    return summary

if __name__ == "__main__":
    preprocess_corpus()
