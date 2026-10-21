import os
import sys
import time
import json
import urllib.request
import urllib.error

BASE_URL = "https://arctic-shift.photon-reddit.com/api"

def download_data(content_type="posts", subreddit="JEENEETards", output_file=None, max_records=None, start_after_ms=None):
    """
    Downloads posts or comments from Arctic Shift directly into a JSONL file.
    Supports auto-resuming from existing files.
    """
    if not output_file:
        os.makedirs("data", exist_ok=True)
        output_file = f"data/{subreddit}_{content_type}.jsonl"

    current_after = start_after_ms

    # Check for existing records to resume
    count = 0
    if os.path.exists(output_file) and current_after is None:
        last_timestamp = None
        print(f"Checking existing file '{output_file}' for resume point...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
                    try:
                        record = json.loads(line)
                        if "created_utc" in record:
                            last_timestamp = record["created_utc"]
                    except json.JSONDecodeError:
                        pass
        if last_timestamp:
            current_after = int(last_timestamp * 1000)
            print(f"Resuming {content_type} from {count} existing records (timestamp {current_after})...")

    print(f"\n[+] Starting download for r/{subreddit} {content_type} -> {output_file}")
    
    headers = {
        "User-Agent": "IRE-SearchEngine-Downloader/1.0"
    }

    import ssl
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_ctx = ssl._create_unverified_context()

    consecutive_errors = 0
    with open(output_file, "a", encoding="utf-8") as out:
        while True:
            params = f"subreddit={subreddit}&limit=auto&sort=asc&meta-app=download-tool"
            if current_after is not None:
                params += f"&after={current_after}"
            
            url = f"{BASE_URL}/{content_type}/search?{params}"
            
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                
                items = payload.get("data", [])
                if not items:
                    print(f"\n[✓] Completed {content_type}! No more records returned.")
                    break
                
                for item in items:
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    if max_records and count >= max_records:
                        print(f"\n[✓] Reached limit of {max_records} records.")
                        return count

                out.flush()
                last_utc = items[-1].get("created_utc")
                if last_utc is not None:
                    next_after = int(last_utc * 1000)
                    if next_after == current_after:
                        next_after += 1000
                    current_after = next_after
                
                sys.stdout.write(f"\rFetched {count:,} {content_type}... (latest: {time.strftime('%Y-%m-%d', time.gmtime(last_utc))})")
                sys.stdout.flush()
                
                consecutive_errors = 0
                time.sleep(0.3)  # Polite rate limiting

            except Exception as e:
                consecutive_errors += 1
                wait_time = min(60, 2 ** consecutive_errors)
                print(f"\n[!] Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                if consecutive_errors > 6:
                    print("\n[X] Too many consecutive errors. Aborting.")
                    break

    print(f"\n[✓] Done. Total {content_type} saved: {count:,}")
    return count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["posts", "comments", "both"], default="posts", help="What to download")
    parser.add_argument("--sub", default="JEENEETards", help="Subreddit name")
    parser.add_argument("--limit", type=int, default=None, help="Max records to download (optional)")
    args = parser.parse_args()

    if args.type in ["posts", "both"]:
        download_data("posts", subreddit=args.sub, max_records=args.limit)
    if args.type in ["comments", "both"]:
        download_data("comments", subreddit=args.sub, max_records=args.limit)
