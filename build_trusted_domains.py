#!/usr/bin/env python3
"""
Builds the trusted domain allowlist from Zenodo news domains,
the Iffy unreliable blocklist, and HF reliability labels.
"""

import argparse
import json
from pathlib import Path
from typing import Set, Tuple

import pandas as pd
import requests
import tldextract
from datasets import load_dataset

ZENODO_NEWS_DOMAINS_CSV = "https://zenodo.org/records/17080910/files/news_domains.csv?download=1"
IFFY_OPENSHEET_JSON = "https://opensheet.elk.sh/1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/Iffy-news"
HF_RELIABILITY_DATASET = "sergioburdisso/news_media_reliability"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

NEWS_DOMAINS_PATH = DATA_DIR / "news_domains.csv"
UNRELIABLE_PATH = DATA_DIR / "unreliable_domains.json"
TRUSTED_CSV_PATH = DATA_DIR / "trusted_domains.csv"
TRUSTED_JSON_PATH = DATA_DIR / "trusted_domains.json"


def registrable_domain(s: str) -> str:
    ext = tldextract.extract((s or "").strip())
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()


def download_file(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def load_news_domain_universe() -> Set[str]:
    if not NEWS_DOMAINS_PATH.exists():
        print(f"Downloading news domain universe from Zenodo -> {NEWS_DOMAINS_PATH}")
        download_file(ZENODO_NEWS_DOMAINS_CSV, NEWS_DOMAINS_PATH)

    df = pd.read_csv(NEWS_DOMAINS_PATH)
    col = "domain" if "domain" in df.columns else df.columns[0]
    domains = set(registrable_domain(x) for x in df[col].astype(str).tolist())
    return {d for d in domains if d}


def load_iffy_unreliable_domains() -> Set[str]:
    print("Downloading unreliable domains (Iffy JSON view)")
    r = requests.get(IFFY_OPENSHEET_JSON, timeout=60)
    r.raise_for_status()
    rows = r.json()

    candidate_keys = ["Domain","domain","Site","site","URL","url","Website","website"]
    unreliable = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        val = None
        for k in candidate_keys:
            if k in row and row[k]:
                val = str(row[k])
                break
        if not val:
            continue
        d = registrable_domain(val)
        if d:
            unreliable.add(d)

    UNRELIABLE_PATH.write_text(
        json.dumps({"source": IFFY_OPENSHEET_JSON, "unreliable_domains": sorted(unreliable)}, indent=2),
        encoding="utf-8"
    )
    return unreliable


def load_hf_reliable_domains(min_newsguard_score: int | None) -> Set[str]:
    print(f"Loading HF dataset: {HF_RELIABILITY_DATASET}")
    ds = load_dataset(HF_RELIABILITY_DATASET, split="train")
    df = ds.to_pandas()

    df["domain_norm"] = df["domain"].astype(str).apply(registrable_domain)
    df = df[df["domain_norm"] != ""]
    df = df[df["reliability_label"] == 1]

    if min_newsguard_score is not None and "newsguard_score" in df.columns:
        df = df[df["newsguard_score"].fillna(-1) >= min_newsguard_score]

    return set(df["domain_norm"].tolist())


def build_trusted(universe: Set[str], unreliable: Set[str], reliable: Set[str], mode: str) -> Tuple[Set[str], dict]:
    if mode == "intersection":
        trusted = (universe & reliable) - unreliable
    elif mode == "union":
        trusted = (reliable | (universe - unreliable)) - unreliable
    else:
        raise ValueError("mode must be: intersection | union")

    meta = {
        "mode": mode,
        "counts": {
            "universe_news_domains": len(universe),
            "iffy_unreliable_domains": len(unreliable),
            "hf_reliable_domains": len(reliable),
            "trusted_domains_final": len(trusted),
        }
    }
    return trusted, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intersection", "union"], default="intersection",
                    help="How to combine lists. Use 'intersection' for stricter trusted allowlist.")
    ap.add_argument("--min-newsguard-score", type=int, default=None,
                    help="Optional: keep only HF reliable domains with newsguard_score >= this value (e.g., 80).")
    args = ap.parse_args()

    universe = load_news_domain_universe()
    unreliable = load_iffy_unreliable_domains()
    reliable = load_hf_reliable_domains(args.min_newsguard_score)

    trusted, meta = build_trusted(universe, unreliable, reliable, args.mode)

    pd.DataFrame({"domain": sorted(trusted)}).to_csv(TRUSTED_CSV_PATH, index=False)

    payload = {
        "trusted_domains": sorted(trusted),
        "meta": meta,
        "sources": {
            "zenodo_news_domains_csv": ZENODO_NEWS_DOMAINS_CSV,
            "iffy_json": IFFY_OPENSHEET_JSON,
            "hf_dataset": HF_RELIABILITY_DATASET,
        }
    }
    TRUSTED_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Trusted domains CSV : {TRUSTED_CSV_PATH}")
    print(f"Trusted domains JSON: {TRUSTED_JSON_PATH}")
    print("Meta:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
