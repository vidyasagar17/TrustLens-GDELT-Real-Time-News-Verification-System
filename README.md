# TrustLens-GDELT

Trust-gated news verification using GDELT and a local LLM.

Given a claim, this system queries GDELT for matching articles, filters them through a trusted domain allowlist built from public datasets, classifies whether each article **supports or refutes** the claim, scores corroboration across independent outlets, and generates a citation-backed report via a local LLaMA model (GGUF via llama.cpp).

The verdict is decided by a deterministic rule engine. The LLM only writes prose about a verdict already settled — it never adjudicates. See [HALLUCINATION_MITIGATION.md](HALLUCINATION_MITIGATION.md) for the threat model and design rationale.

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

The LLM is only invoked when the rule engine reaches a directional verdict. It receives the settled verdict plus sanitized evidence, and its output is rejected outright if any bullet is uncited or cites an item that does not exist.

### Stance detection

An NLI model classifies each trusted article as `SUPPORTS`, `REFUTES`, or `DISCUSSES` the claim. This is what separates *"reputable outlets are covering these words"* from *"reputable outlets assert this is true"* — without it, a widely-debunked claim scores highest, because debunkings are themselves trusted-outlet coverage.

Enabled by default; set `STANCE_ENABLED=0` to fall back to coverage-only scoring. Override the model with `STANCE_MODEL` (default `microsoft/deberta-v3-base-mnli`, ~180MB, CPU-friendly).

With stance detection, the system can express verdicts the coverage-only pipeline structurally cannot: `REFUTED`, `LIKELY_REFUTED`, and `DISPUTED`.

## Evaluation

```bash
python eval/evaluate.py                 # seed set; baseline vs stance, side by side
python eval/evaluate.py --sweep         # tune verdict thresholds on cached results
python eval/evaluate.py --dataset averitec.json --limit 200
```

Reports accuracy, per-class precision/recall/F1, a confusion matrix, and the **critical failure rate** — the share of false claims returned as `SUPPORTED`. For a verification system that is the metric that matters; overall accuracy can hide it completely.

GDELT responses are cached to `eval/.cache/`, so reruns and threshold sweeps cost no extra API calls.

`eval/seed_claims.json` is a hand-authored 12-claim smoke test, **not a benchmark** — it exists so the harness runs on day one. For real numbers, point `--dataset` at [AVeriTeC](https://fever.ai/dataset/averitec.html), FEVER, or LIAR; the loader maps their label vocabularies automatically.

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
