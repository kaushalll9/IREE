"""
Phase 4: IR Evaluation & Benchmark Suite for IRE Project
Computes Precision@K, Recall@K, MAP, MRR, and NDCG@K across multiple ranking configurations.
"""

import os
import math
import json
import time
from typing import Dict, List, Any
from ranker import BM25SearchEngine

# Standardized Information Needs and Ground Truth Relevance Judgments (Qrels)
# Judged by pooling (python evaluate.py --pool): the union of every configuration's
# top-10 per query was assessed against the stated intent from title + body.
# Graded relevance: 2 = Highly Relevant, 1 = Partially Relevant, 0 = Irrelevant
BENCHMARK_QUERIES = [
    {
        "query_id": "Q1",
        "query": "physics wallah vs unacademy",
        "intent": "Comparison between Physics Wallah and Unacademy platforms for JEE/NEET preparation",
        "qrels": {
            "qttdqp": 2,  # need advice (phy wallah / unacademy)
            "vcnzuw": 2,  # Urgent advice
            "u13nyh": 2,  # Physics Wallah vs ATP Star vs Unacademy vs Vedantu vs Allen digital
            "vcqn3x": 2,  # Physics Wallah or Unacademy ??
            "vgcmru": 2,  # Allen / Unacademy / Reliable / Physics wallah which to choose?
            "ueh9xe": 2,  # Which is better for 11th class student physics wallah vs un vs YouTube
            "wdjrrg": 2,  # Unacademy Atoms or JEE Wallah
            "u6enai": 1,  # Physics Wallah for Advanced?
            "sepacn": 1,  # Physics Wallah or Physics Galaxy for Neet physics?
            "uwttyl": 1,  # Physics Wallah final reply to Unacademy
            "p6vuaf": 1,  # jee wallah vs physics galaxy checkilst
            "wbyki5": 1,  # khan sir vs phyjics wallah
            "ntfsp2": 1,  # Arjuna Batch -PW
            "s4imhw": 0,  # Why is Physics Wallah hated?
            "rj2dsg": 0,  # GOC Advice Please
            "tq4ghv": 0,  # Are Pogi, Physics Wallah, Unacademy, Mohit Tyagi Memes actually funny?
            "uytn1f": 0,  # Physics Wallah
            "ujwu33": 0  # Mhmm Physics Wallah
        }
    },
    {
        "query_id": "Q2",
        "query": "best book for organic chemistry",
        "intent": "Recommendations for JEE/NEET organic chemistry theory and problem-solving books",
        "qrels": {
            "v37tm3": 2,  # best book for organic chemistry theory
            "wdj4oc": 2,  # Organic Chemistry
            "u7y6za": 2,  # Best objective book for Chemistry (Physical, Organic, Inorganic) to Cr
            "m9nluq": 2,  # Book for Organic chemistry
            "u8b099": 2,  # ORGANIC CHEMISTRY (JEE)
            "w4jq5w": 2,  # Replica Blaci book for JEE or MS Chouhan Sir book for advance for orga
            "t32r04": 2,  # SUGGEST ORGANIC CHEMISTRY THEORY BOOK
            "ufb2w3": 2,  # Which is the best reference book for organic chemistry neet?
            "qieg1k": 2,  # Book recommendations for organic chemistry?
            "w0g7ye": 1,  # Organic chemistry
            "wai6fl": 1,  # Best Books for Neet+Boards
            "qgaqen": 1,  # Best way to learn Organic chemistry from scratch?
            "pz97fm": 0,  # Organic chemistry
            "o7teuq": 0  # Best book for physical chemistry?
        }
    },
    {
        "query_id": "Q3",
        "query": "marks vs percentile 2022",
        "intent": "Analysis or questions regarding marks to percentile conversion for JEE Main 2022",
        "qrels": {
            "wbr1xm": 2,  # Drop in Marks vs Percentile vs Rank
            "vl6p8q": 2,  # marks vs percentile
            "ubhftk": 2,  # Those who know about percentile vs marks
            "vwdiza": 2,  # marks vs percentile
            "we9n1g": 2,  # session 1 marks vs percentile
            "wcnjdd": 2,  # Tit le
            "vw9vlu": 2,  # 28 S2 ka crazy Marks vs Percentile
            "w177lb": 2,  # marks vs percentile
            "v5dbyp": 2,  # What is going to be the expected marks vs percentile this year? {JEE M
            "visyen": 2,  # Can someone give me a legit "marks vs percentile sheet" for JEE 2022?
            "vpm9c3": 1,  # jee mains 2022 percentile vs rank
            "vvzt4o": 1  # MARKS VS PERCENTILE batao
        }
    },
    {
        "query_id": "Q4",
        "query": "allen test series review",
        "intent": "Reviews, feedback, or difficulty comparison of Allen leader/major test series",
        "qrels": {
            "twr9n9": 2,  # How's Allen leader test series for class 12.
            "tpp7hw": 2,  # ALLEN TEST SERIES (JEE)
            "ugt82d": 2,  # Allen Test Series NEET
            "rxlwkp": 2,  # Wanted to ask about this allen online test series...
            "sx1yuh": 2,  # Has anyone taken this allen leader test series or heard anything about
            "vmahre": 1,  # ALLEN TEST SERIES
            "sup6pu": 1,  # Allen
            "w68d98": 1,  # How many ppl do u think is does Allen test series for adv?
            "v8boga": 0,  # Mathongo Test Series Review for JEEM 2022
            "sutrj1": 0,  # Allen 2023 test series
            "svf1k6": 0,  # Test Series
            "ur3o3j": 0,  # ALLEN TEST SERIES
            "w3rf1t": 0,  # Allen Test Series
            "tq8k0y": 0  # URGENT REVIEW BOIZZ......... Mathongo jee adv test series
        }
    },
    {
        "query_id": "Q5",
        "query": "dc pandey vs hc verma",
        "intent": "Comparison between DC Pandey and HC Verma physics books for JEE physics concepts and problems",
        "qrels": {
            "uvyblk": 2,  # DC Pandey VS HC VERMA
            "ur0zb7": 2,  # DC Pandey vs HC Verma for theory
            "tyejuf": 2,  # Weak in physics. After doing hc Verma should I refer to other books li
            "s6415s": 2,  # Hcv or dc pandey for neet
            "pf71kf": 1,  # Does HC Verma skip over a lot of theory?
            "vmr2cz": 1,  # Is HC Verma theory sufficient for 12th and JEE
            "u3lz9l": 1,  # HC VERMA FOR NEET
            "kfmr9h": 1,  # Do we need to solve HC Verma for NEET 2021 to get a 150+ score?
            "olykg5": 1,  # Is hc verma enough for jee mains
            "vf1n3t": 1,  # Should I solve HC Verma if I am preparing for NEET ?
            "qoqeca": 1,  # Should I go for DC Pandey?
            "rlfdby": 1,  # How is hc verma book
            "tpghz8": 0,  # Are PG Books Enough for JEE Mains & Advanced level Physics?
            "v3w8ib": 0,  # Study sequence for a chapter in physics
            "qq2jx0": 0,  # Hc Verma win padmashree ,alakh pandey sir win ballon d'or
            "sxpizw": 0  # Not Able to Solve HC Verma!
        }
    },
    {
        "query_id": "Q6",
        "query": "maths dropper strategy",
        "intent": "Advice and study schedules for droppers struggling with or targeting JEE Mathematics",
        "qrels": {
            "oobtq9": 2,  # Dropper jee 2022 not being able to do maths problems
            "ttksmg": 1,  # Make a Dropper strategy
            "ok16c4": 1,  # JEE 2022 droppers assemble.
            "w4fu3t": 1,  # droppers strategy required
            "wa4gol": 1,  # 2023
            "vv012n": 1,  # Strategy for JEE Main maths
            "vpsx4j": 1,  # Jee Main 2022 Maths
            "vl4ikh": 1,  # JEE droppers for 23, what's your strategy/approach for the upcoming ye
            "we11e9": 1,  # dropper maths frim baeics
            "sbjuqc": 0,  # Block Strategy UwU
            "t4tf1z": 0,  # Doubt regarding block strategy from Ashish sir
            "w5rzwo": 0  # Not a Strategy Video 🗿🗿🗿
        }
    },
    {
        "query_id": "Q7",
        "query": "neet 600 marks government medical college",
        "intent": "Possibility and cutoffs for securing a government MBBS seat with 600 marks in NEET",
        "qrels": {
            "u7vh7a": 2,  # How many seats in government medical colleges for general category stu
            "sqp487": 2,  # Need NEET help please
            "sd0clv": 2,  # Does anyone over here know how much one needs to generally score in NE
            "w25s5m": 2,  # Neet 600+ marks pe kya milega
            "w2wqqs": 2,  # How much marks & rank in NEET is required for any government college i
            "w6sk91": 1,  # Kuchch ho sakta hai kya ?
            "ts2sbg": 1,  # NEET medical colleges cutoff
            "tyw36d": 1,  # How hard is it to get into a government medical college?
            "vl3t50": 1,  # 600+ NEET (KMC MANIPAL)
            "sgr73e": 0,  # Neet 2022 450 marks possible?
            "rnnnmx": 0,  # AIIMS AND ALL MEDICAl KALEJ are FAKE!!!!!!!!!!!!!!!!!!!! (With Proof)
            "w1tu9w": 0,  # private medical colleges
            "v348x9": 0  # How was medical college entrance before NEET?
        }
    },
    {
        "query_id": "Q8",
        "query": "mock test score stuck",
        "intent": "Advice for students whose mock scores plateaued or are not improving",
        "qrels": {
            "sq2b10": 2,  # Mock test score is not improving
            "ufe885": 2,  # Stuck at 150
            "uum5bu": 2,  # Jeetards scoring 150+ in mocks, how to improve?
            "vyvm0g": 1,  # how much are you guyz scoring in mocks for neet
            "puep0g": 1,  # Mock test are messing me up!
            "vdgih5": 1,  # For those who've taken ALLEN's BITSAT Mock test
            "rrrfmi": 0,  # NTA Mock test scores
            "u4yga7": 0,  # NTA MOCK TEST
            "rrq1oq": 0,  # Mock Test Scores??
            "v053zw": 0,  # Did anyone start giving mocks for Bitsat??
            "qg6h6b": 0,  # What score are you getting in kvpy mock tests
            "qg80uw": 0,  # What score are you getting in kvpy mock tests.
            "syfic3": 0,  # JEE 22 Mock test scores
            "vy24c2": 0  # Is 70 a good score by random marking in JEE mock tests?
        }
    },
    {
        "query_id": "Q9",
        "query": "hostel food kota life",
        "intent": "Discussions, reality checks, and experiences with hostel food and life in Kota",
        "qrels": {
            "ifib7f": 2,  # How was your hostel food?
            "ifi9fm": 2,  # Rate your hostel food out of 10?
            "rciayu": 2,  # Avg kota Hostel (I was not in the video)
            "wed70a": 2,  # Hostel life day 2 Update
            "wdayhx": 1,  # hostel guide..
            "qye2ux": 1,  # Hostel Life
            "w46rft": 1,  # Which is the best hostel in Kota?
            "wcpa8f": 1,  # Hostel survival tips!!
            "r8lz7r": 1,  # STUDY SPACE @KOTA HOSTEL
            "vqanx6": 1,  # my Luxury Hostel back when I was in Kota 🥲
            "vqao0c": 1,  # my Luxury Hostel back when I was in Kota 🥲
            "vrvo4c": 0,  # Kota query?idk title
            "sal2vv": 0,  # Hidden Kota hostel talent show
            "qf94lr": 0  # Kota Hostels After yesterday's match :) (p.s. they are not celebrating
        }
    },
    {
        "query_id": "Q10",
        "query": "iit bombay cse cutoff rank",
        "intent": "Cutoff ranks, closing ranks, and percentiles required for Computer Science at IIT Bombay",
        "qrels": {
            "s6xfhw": 2,  # How to know the rank required for IIT/NIT?
            "prubmb": 2,  # IIT cutoff list?
            "q9uwyt": 1,  # Ok gays I might have a chance of bombay cse
            "vv4mil": 0,  # IIT BOMBAY CSE pakka ab 🗿
            "w0m8bu": 0,  # BITS HYD CSE
            "wco4b2": 0,  # MIT CUTOFF
            "ir8efd": 0,  # siRiUs query about IIT bombay
            "vvpy6m": 0,  # SRMJEEE best of all phases ?
            "o93j5t": 0,  # How can I get into IIT Bombay in these last few months?
            "q4cmmm": 0,  # If you got the chance i.e. All India Rank 2000. Would you choose IIT K
            "tc5bf6": 0,  # Why??
            "n6u296": 0,  # IIT Bombay here I come
            "qq1t2v": 0,  # IIT BOMBAY MOTIVASUN 🥵
            "qq1t2y": 0,  # IIT BOMBAY MOTIVASUN 🥵
            "txrjzs": 0,  # IIT Bombay I'm cuming for u😈
            "vpkafr": 0  # IIT Bombay Motivation
        }
    }
]

def compute_dcg(relevances: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def compute_ndcg(relevances: List[int], ideal_relevances: List[int], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg(relevances, k)
    idcg = compute_dcg(sorted(ideal_relevances, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def evaluate_configuration(engine: BM25SearchEngine, config_name: str, config_kwargs: dict) -> Dict[str, Any]:
    """Runs all benchmark queries through a specific ranking configuration and calculates IR metrics."""
    t0 = time.time()
    
    per_query_results = []
    
    p1_list = []
    p3_list = []
    p5_list = []
    p10_list = []
    ap_list = []
    rr_list = []
    ndcg5_list = []
    ndcg10_list = []
    latencies = []
    
    for item in BENCHMARK_QUERIES:
        query = item["query"]
        qrels = item["qrels"]
        
        # Execute query
        res = engine.search(query=query, top_k=10, **config_kwargs)
        latencies.append(res["latency_ms"])
        
        retrieved_ids = [doc["doc_id"] for doc in res["results"]]
        
        # Binary & graded relevance vectors
        binary_rel = [1 if qrels.get(did, 0) > 0 else 0 for did in retrieved_ids]
        graded_rel = [qrels.get(did, 0) for did in retrieved_ids]
        all_qrel_values = list(qrels.values())
        
        # P@K
        p1 = binary_rel[0] if len(binary_rel) >= 1 else 0.0
        p3 = sum(binary_rel[:3]) / 3.0 if len(binary_rel) >= 3 else sum(binary_rel) / 3.0
        p5 = sum(binary_rel[:5]) / 5.0 if len(binary_rel) >= 5 else sum(binary_rel) / 5.0
        p10 = sum(binary_rel[:10]) / 10.0 if len(binary_rel) >= 10 else sum(binary_rel) / 10.0
        
        # Reciprocal Rank (RR)
        rr = 0.0
        for rank_idx, rel in enumerate(binary_rel, start=1):
            if rel > 0:
                rr = 1.0 / rank_idx
                break
                
        # Average Precision (AP)
        num_relevant_found = 0
        sum_precisions = 0.0
        for rank_idx, rel in enumerate(binary_rel, start=1):
            if rel > 0:
                num_relevant_found += 1
                sum_precisions += (num_relevant_found / rank_idx)
                
        total_relevant = len([v for v in all_qrel_values if v > 0])
        ap = sum_precisions / total_relevant if total_relevant > 0 else 0.0
        
        # NDCG@5 & NDCG@10
        ndcg5 = compute_ndcg(graded_rel, all_qrel_values, k=5)
        ndcg10 = compute_ndcg(graded_rel, all_qrel_values, k=10)
        
        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)
        p10_list.append(p10)
        rr_list.append(rr)
        ap_list.append(ap)
        ndcg5_list.append(ndcg5)
        ndcg10_list.append(ndcg10)
        
        per_query_results.append({
            "query_id": item["query_id"],
            "query": query,
            "p@1": round(p1, 4),
            "p@5": round(p5, 4),
            "p@10": round(p10, 4),
            "rr": round(rr, 4),
            "ap": round(ap, 4),
            "ndcg@5": round(ndcg5, 4),
            "ndcg@10": round(ndcg10, 4),
            "latency_ms": res["latency_ms"]
        })
        
    num_q = len(BENCHMARK_QUERIES)
    summary = {
        "config_name": config_name,
        "mean_p@1": round(sum(p1_list) / num_q, 4),
        "mean_p@3": round(sum(p3_list) / num_q, 4),
        "mean_p@5": round(sum(p5_list) / num_q, 4),
        "mean_p@10": round(sum(p10_list) / num_q, 4),
        "mrr": round(sum(rr_list) / num_q, 4),
        "map": round(sum(ap_list) / num_q, 4),
        "ndcg@5": round(sum(ndcg5_list) / num_q, 4),
        "ndcg@10": round(sum(ndcg10_list) / num_q, 4),
        "avg_latency_ms": round(sum(latencies) / num_q, 2),
        "per_query": per_query_results
    }
    return summary

# Define 4 configurations for ablation study:
CONFIGURATIONS = [
    {
        "name": "Standard BM25 (Unfielded)",
        "kwargs": {
            "k1": 1.2,
            "w_title": 1.0,
            "w_body": 1.0,
            "use_engagement_boost": False
        }
    },
    {
        "name": "BM25F (Field-Weighted: Title 2.5x)",
        "kwargs": {
            "k1": 1.2,
            "w_title": 2.5,
            "w_body": 1.0,
            "use_engagement_boost": False
        }
    },
    {
        "name": "BM25F + Engagement Boosting",
        "kwargs": {
            "k1": 1.2,
            "w_title": 2.5,
            "w_body": 1.0,
            "use_engagement_boost": True
        }
    },
    {
        "name": "BM25F + Engagement + Domain Expansion (Proposed Model)",
        "kwargs": {
            "k1": 1.2,
            "w_title": 2.5,
            "w_body": 1.0,
            "use_engagement_boost": True,
            "expand": True
        }
    }
]



def validate_qrels(engine: BM25SearchEngine):
    """Every judged doc must exist in the index, or the metrics are silently wrong."""
    known = {m["doc_id"] for m in engine.metadata}
    missing = [(q["query_id"], d) for q in BENCHMARK_QUERIES for d in q["qrels"] if d not in known]
    if missing:
        raise ValueError(f"Qrels reference {len(missing)} doc_ids not in the corpus: {missing}")


def print_pool(engine: BM25SearchEngine, depth: int = 10):
    """
    Pooling (TREC-style): union of the top-`depth` results of every configuration.
    Prints each pooled doc with its current judgment ('?' = unjudged) so the
    pool can be assessed and added to BENCHMARK_QUERIES.
    """
    for item in BENCHMARK_QUERIES:
        pool = {}
        for cfg in CONFIGURATIONS:
            for doc in engine.search(query=item["query"], top_k=depth, **cfg["kwargs"])["results"]:
                pool.setdefault(doc["doc_id"], doc)
        print(f"\n## {item['query_id']} {item['query']!r} - {item['intent']}")
        for did, doc in pool.items():
            grade = item["qrels"].get(did, "?")
            print(f"  {grade} {did} | {doc['title'][:90]} | {doc['snippet'][:120]}")

def run_evaluation_suite(index_dir: str = "index", output_dir: str = "evaluation"):
    """Runs comparative evaluation across 4 ranking configurations and writes benchmark reports."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 75)
    print("🔬 Starting Phase 4: Information Retrieval Evaluation & Benchmarking")
    print("=" * 75)
    
    engine = BM25SearchEngine(index_dir=index_dir)
    engine.load_index()
    
    validate_qrels(engine)

    results = []
    for cfg in CONFIGURATIONS:
        print(f"\n[+] Evaluating Model: {cfg['name']}...")
        summary = evaluate_configuration(engine, cfg["name"], cfg["kwargs"])
        results.append(summary)
        print(f"    MAP: {summary['map']} | MRR: {summary['mrr']} | NDCG@10: {summary['ndcg@10']} | Latency: {summary['avg_latency_ms']} ms")

    # Save JSON results
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_queries_count": len(BENCHMARK_QUERIES),
            "benchmark_queries": BENCHMARK_QUERIES,
            "models_evaluated": results
        }, f, indent=2)

    # Generate Markdown Report
    md_path = os.path.join(output_dir, "evaluation_report.md")
    generate_markdown_report(results, md_path)
    
    # Print Terminal Comparison Table
    print("\n" + "=" * 75)
    print("📊 Phase 4 IR Evaluation Benchmark Comparison Table")
    print("=" * 75)
    header = f"{'Ranking Configuration':<42} | {'P@1':<6} | {'P@5':<6} | {'MRR':<6} | {'MAP':<6} | {'NDCG@10':<7} | {'Latency':<8}"
    print(header)
    print("-" * len(header))
    for res in results:
        row = f"{res['config_name']:<42} | {res['mean_p@1']:<6.4f} | {res['mean_p@5']:<6.4f} | {res['mrr']:<6.4f} | {res['map']:<6.4f} | {res['ndcg@10']:<7.4f} | {res['avg_latency_ms']:<5.2f} ms"
        print(row)
    print("=" * 75)
    print(f"\n[✓] Detailed evaluation reports saved to:")
    print(f"    - {json_path}")
    print(f"    - {md_path}")
    return results

def generate_markdown_report(results: List[Dict[str, Any]], filepath: str):
    lines = []
    lines.append("# Information Retrieval Evaluation Report (Phase 4)")
    lines.append("\n**Course:** Information Retrieval and Extraction (IRE) Project  ")
    lines.append(f"**Dataset:** `r/JEENEETards` Cleaned Corpus (43,116 documents)  ")
    lines.append(f"**Test Set:** {len(BENCHMARK_QUERIES)} Representative Domain Information Needs  ")
    lines.append("\n---\n")
    lines.append("## 1. Executive Summary & Model Comparison\n")
    lines.append("| Ranking Model | P@1 | P@5 | P@10 | MRR | MAP | NDCG@10 | Avg Latency |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        lines.append(f"| **{r['config_name']}** | {r['mean_p@1']} | {r['mean_p@5']} | {r['mean_p@10']} | {r['mrr']} | {r['map']} | {r['ndcg@10']} | {r['avg_latency_ms']} ms |")
    
    lines.append("\n### Key Observations (computed from this run):")
    for label, key in [("MAP", "map"), ("NDCG@10", "ndcg@10"), ("P@1", "mean_p@1"), ("P@5", "mean_p@5")]:
        best = max(results, key=lambda r: r[key])
        lines.append(f"- **Best {label}**: {best['config_name']} ({best[key]})")
    lines.append(f"- **Latency**: {min(r['avg_latency_ms'] for r in results)}-{max(r['avg_latency_ms'] for r in results)} ms per query across all configurations.")
    lines.append("- With only 10 queries, differences of a few points are within noise; treat them as indicative, not significant.")
    lines.append("\n---\n")
    lines.append("## 2. Per-Query Breakdown (Proposed Model: BM25F + Engagement + Domain Expansion)\n")
    lines.append("| Query ID | Query | P@1 | P@5 | MRR | MAP | NDCG@10 | Latency |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    best_model = results[-1]
    for pq in best_model["per_query"]:
        lines.append(f"| **{pq['query_id']}** | `{pq['query']}` | {pq['p@1']} | {pq['p@5']} | {pq['rr']} | {pq['ap']} | {pq['ndcg@10']} | {pq['latency_ms']} ms |")
        
    lines.append("\n---\n")
    lines.append("## 3. Evaluation Methodology")
    lines.append("- **Test Queries**: 10 hand-curated queries covering Coaching Comparisons, Book Suggestions, Score Cutoffs, Dropper Strategies, Test Series, and Mental Wellbeing.")
    lines.append("- **Metrics Defined**: Precision@K ($K=1, 3, 5, 10$), Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (NDCG@10).")
    lines.append("- **Relevance Assessments (Qrels)**: Graded 0 (Irrelevant), 1 (Partially Relevant), 2 (Highly Relevant).")
    lines.append("- **Pooling**: judgments cover the union of every configuration's top-10 per query (`python evaluate.py --pool`), so each evaluated ranking is fully judged in its top 10. Unjudged documents count as irrelevant.")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="print the judgment pool instead of evaluating")
    args = ap.parse_args()
    if args.pool:
        e = BM25SearchEngine()
        e.load_index()
        print_pool(e)
    else:
        run_evaluation_suite()
