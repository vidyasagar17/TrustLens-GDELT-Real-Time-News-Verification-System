#!/usr/bin/env python3
"""
TrustLens evaluation harness.

Answers the question the project could not previously answer: does it work?

Reports, per configuration:
  - accuracy / per-class precision, recall, F1
  - a confusion matrix
  - CRITICAL FAILURE RATE: false claims returned as SUPPORTED. For a verification
    system this is the metric that matters; overall accuracy can hide it entirely.
  - coverage-only vs stance-based verdicts side by side, which is the direct
    measurement of what stance detection bought.

GDELT responses are cached to eval/.cache/ so reruns are offline and fast, and so
a threshold sweep costs zero extra API calls.

Usage:
    python eval/evaluate.py                        # seed set, both configs
    python eval/evaluate.py --sweep                # tune thresholds on cached data
    python eval/evaluate.py --dataset averitec.json --limit 200
    python eval/evaluate.py --no-stance            # coverage-only baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdelt_client import search_articles  # noqa: E402
from trust_policy import load_trusted_domains, load_unreliable_domains  # noqa: E402
from pipeline import run_verification  # noqa: E402
import verifier  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / ".cache"
SEED_SET = EVAL_DIR / "seed_claims.json"

# Hedged verdicts collapse onto their firm counterpart for scoring; DISPUTED is
# treated as a non-answer, same as abstention.
COLLAPSE = {
    "SUPPORTED": "SUPPORTED",
    "LIKELY_SUPPORTED": "SUPPORTED",
    "REFUTED": "REFUTED",
    "LIKELY_REFUTED": "REFUTED",
    "DISPUTED": "INSUFFICIENT_EVIDENCE",
    "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
}
CLASSES = ["SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE"]


# --- cached retrieval -----------------------------------------------------------

def cached_search(query: str, max_records: int = 50) -> List[Dict]:
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.sha256(f"{query}|{max_records}".encode("utf-8")).hexdigest()[:24]
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        articles = search_articles(query, max_records=max_records)
    except Exception as e:
        print(f"  ! retrieval failed for {query!r}: {type(e).__name__}: {e}")
        articles = []
    path.write_text(json.dumps(articles), encoding="utf-8")
    time.sleep(1.0)  # be polite to the GDELT API
    return articles


# --- dataset loading ------------------------------------------------------------

def load_dataset(path: Path, limit: Optional[int] = None) -> List[Dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    records = obj["claims"] if isinstance(obj, dict) else obj

    normalized = []
    for r in records:
        claim = r.get("claim") or r.get("statement") or r.get("text")
        label = (r.get("label") or r.get("gold") or "").strip().upper()
        # Map common benchmark vocabularies onto ours.
        label = {
            "TRUE": "SUPPORTED",
            "SUPPORTS": "SUPPORTED",
            "FALSE": "REFUTED",
            "REFUTES": "REFUTED",
            "NOT ENOUGH INFO": "INSUFFICIENT_EVIDENCE",
            "NEI": "INSUFFICIENT_EVIDENCE",
            "CONFLICTING": "INSUFFICIENT_EVIDENCE",
        }.get(label, label)
        if claim and label in CLASSES:
            normalized.append({**r, "claim": claim, "label": label})
    return normalized[:limit] if limit else normalized


# --- metrics --------------------------------------------------------------------

def compute_metrics(rows: List[Dict], pred_key: str) -> Dict:
    gold = [r["label"] for r in rows]
    pred = [COLLAPSE.get(r[pred_key], "INSUFFICIENT_EVIDENCE") for r in rows]

    confusion: Dict[str, Counter] = defaultdict(Counter)
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    per_class = {}
    for c in CLASSES:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": sum(1 for g in gold if g == c),
        }

    n_false = sum(1 for g in gold if g == "REFUTED")
    critical = sum(1 for g, p in zip(gold, pred) if g == "REFUTED" and p == "SUPPORTED")

    return {
        "n": len(rows),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(rows), 3) if rows else 0.0,
        "per_class": per_class,
        "confusion": {g: dict(c) for g, c in confusion.items()},
        "critical_failures": critical,
        "critical_failure_rate": round(critical / n_false, 3) if n_false else None,
        "abstention_rate": round(
            sum(1 for p in pred if p == "INSUFFICIENT_EVIDENCE") / len(rows), 3
        ) if rows else 0.0,
    }


def print_report(title: str, m: Dict) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    print(f"  n = {m['n']}    accuracy = {m['accuracy']}    abstention = {m['abstention_rate']}")

    cfr = m["critical_failure_rate"]
    if cfr is not None:
        flag = "  <-- false claims returned as SUPPORTED" if m["critical_failures"] else ""
        print(f"  CRITICAL FAILURE RATE = {cfr}  ({m['critical_failures']} claims){flag}")

    print(f"\n  {'class':<24} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>5}")
    for c, s in m["per_class"].items():
        print(f"  {c:<24} {s['precision']:>6} {s['recall']:>6} {s['f1']:>6} {s['support']:>5}")

    print("\n  confusion (gold -> predicted):")
    for g, preds in m["confusion"].items():
        print(f"    {g:<24} {dict(preds)}")


# --- runners --------------------------------------------------------------------

def run_dataset(records, trusted, unreliable, stance, max_records) -> List[Dict]:
    rows = []
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['claim'][:66]}")
        out = run_verification(
            claim=rec["claim"],
            trusted_domains=trusted,
            unreliable_domains=unreliable,
            max_records=max_records,
            stance_classifier=stance,
            llm=None,  # verdict quality is independent of the summary layer
            search_fn=cached_search,
        )
        rows.append({
            "claim": rec["claim"],
            "label": rec["label"],
            "synthetic_control": rec.get("synthetic_control", False),
            "verdict": out["verdict"],
            "verdict_coverage_based": out["verdict_coverage_based"],
            "unique_trusted_sources": out["unique_trusted_sources"],
            "stance": out.get("stance"),
            "signals": out.get("signals"),
        })
    return rows


def sweep(rows: List[Dict]) -> None:
    """Re-derive verdicts from cached stance counts under different thresholds.

    The 3-and-2 cutoffs in verifier.py were never validated. This picks them on
    evidence instead, at zero additional API cost.
    """
    print(f"\n{'=' * 68}\nTHRESHOLD SWEEP\n{'=' * 68}")
    if not any(r.get("stance") for r in rows):
        print("  (no stance data -- rerun without --no-stance)")
        return

    print(f"  {'support':>7} {'likely':>7} {'dominance':>10} {'acc':>7} {'crit':>7} {'abstain':>8}")
    best = None
    for support_t in (2, 3, 4, 5):
        for likely_t in (1, 2, 3):
            if likely_t > support_t:
                continue
            for dom in (2.0, 3.0, 5.0):
                orig = (verifier.SUPPORT_THRESHOLD, verifier.LIKELY_THRESHOLD, verifier.DOMINANCE_RATIO)
                verifier.SUPPORT_THRESHOLD = support_t
                verifier.LIKELY_THRESHOLD = likely_t
                verifier.DOMINANCE_RATIO = dom
                try:
                    scored = [
                        {**r, "verdict": verifier.verdict_from_stance(
                            r["stance"]["support_count"], r["stance"]["refute_count"]
                        )}
                        for r in rows if r.get("stance")
                    ]
                    m = compute_metrics(scored, "verdict")
                finally:
                    (verifier.SUPPORT_THRESHOLD,
                     verifier.LIKELY_THRESHOLD,
                     verifier.DOMINANCE_RATIO) = orig

                cfr = m["critical_failure_rate"] or 0.0
                print(f"  {support_t:>7} {likely_t:>7} {dom:>10} {m['accuracy']:>7} "
                      f"{cfr:>7} {m['abstention_rate']:>8}")
                # Prefer configurations that never call a false claim true.
                key = (-cfr, m["accuracy"], -m["abstention_rate"])
                if best is None or key > best[0]:
                    best = (key, (support_t, likely_t, dom))

    if best:
        s, l, d = best[1]
        print(f"\n  best: SUPPORT_THRESHOLD={s} LIKELY_THRESHOLD={l} DOMINANCE_RATIO={d}")
        print("  (chosen by: minimise critical failures, then maximise accuracy)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=SEED_SET)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-records", type=int, default=50)
    ap.add_argument("--no-stance", action="store_true", help="coverage-only baseline")
    ap.add_argument("--sweep", action="store_true", help="tune thresholds on cached results")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results.json")
    args = ap.parse_args()

    records = load_dataset(args.dataset, args.limit)
    if not records:
        print(f"No usable records in {args.dataset}")
        return 1
    print(f"Loaded {len(records)} claims from {args.dataset.name}")

    stance = None
    if not args.no_stance:
        from stance import StanceClassifier
        stance = StanceClassifier()
        if not stance.available:
            print(f"WARNING: stance model unavailable ({stance.load_error}); "
                  f"falling back to coverage-only.")
            stance = None

    trusted = load_trusted_domains()
    unreliable = load_unreliable_domains()
    print(f"Allowlist: {len(trusted)} trusted / {len(unreliable)} unreliable domains\n")

    rows = run_dataset(records, trusted, unreliable, stance, args.max_records)

    # The headline comparison: what stance detection actually bought.
    print_report("BASELINE -- coverage only (counts topic mentions)",
                 compute_metrics(rows, "verdict_coverage_based"))
    if stance:
        print_report("WITH STANCE DETECTION (counts supporting vs refuting sources)",
                     compute_metrics(rows, "verdict"))

    if args.sweep:
        sweep(rows)

    args.out.write_text(json.dumps({
        "dataset": str(args.dataset),
        "n": len(rows),
        "coverage_only": compute_metrics(rows, "verdict_coverage_based"),
        "with_stance": compute_metrics(rows, "verdict") if stance else None,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
