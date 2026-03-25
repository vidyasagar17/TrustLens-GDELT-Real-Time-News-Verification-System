# TrustLens-GDELT

Trust-gated news verification using GDELT and a local LLM.

Given a claim, this system queries GDELT for matching articles, filters them through a trusted domain allowlist built from public datasets, scores corroboration across independent outlets, and optionally generates a citation-backed verification report via a local LLaMA model (GGUF via llama.cpp).

## Data sources

- **News domain universe**: [Zenodo news_domains.csv](https://zenodo.org/records/17080910/files/news_domains.csv?download=1)
- **Unreliable domain blocklist**: [Iffy Index](https://opensheet.elk.sh/1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/Iffy-news)
- **Reliability labels**: [HF: sergioburdisso/news_media_reliability](https://huggingface.co/datasets/sergioburdisso/news_media_reliability)
- **Article search**: [GDELT Doc API](https://api.gdeltproject.org/api/v2/doc/doc)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build the trusted domain allowlist

```bash
python build_trusted_domains.py --mode intersection
# stricter filtering:
python build_trusted_domains.py --mode intersection --min-newsguard-score 80
```

This produces `data/trusted_domains.csv`, `data/trusted_domains.json`, and `data/unreliable_domains.json`.

### Local LLM (optional)

Download any instruct GGUF model and point to it:

```bash
export LLM_MODEL_PATH="models/your_model.gguf"
export LLM_N_CTX="4096"
export LLM_GPU_LAYERS="0"
```

The LLM is only invoked when at least 2 independent trusted sources are found. It receives only article metadata (title/domain/date/URL), not full text.

## Run

```bash
uvicorn app:app --reload
```

Swagger docs at http://127.0.0.1:8000/docs.

```bash
curl -X POST "http://127.0.0.1:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{"claim":"Apple announced a new iPhone model", "max_records": 50}'
```
