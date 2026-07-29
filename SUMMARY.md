# TrustLens-GDELT — Code Summary

A fact-checking API that verifies news claims against corroborating evidence
from trusted sources.

## The big picture

You send it a claim (e.g. *"Apple announced a new iPhone model"*), and it:

1. Searches GDELT's global news index for matching articles
2. Filters to only articles from a pre-vetted list of trustworthy domains
3. Scores how many *independent* trusted outlets corroborate the claim
4. If there's enough corroboration, asks a locally-run LLM to write a
   citation-backed verdict — using **only** the retrieved evidence, not its
   own training knowledge

It's built as a FastAPI web service with one endpoint: `POST /verify`.

## File-by-file

### `build_trusted_domains.py` — offline setup script (run once, before the app)

Builds the "trusted domain allowlist" that the whole system relies on, from
three public sources:

- **Zenodo `news_domains.csv`** — a broad universe of known news domains
- **Iffy Index** (via an opensheet.elk.sh JSON proxy of a Google Sheet) — a
  blocklist of unreliable/misinformation domains
- **HuggingFace `sergioburdisso/news_media_reliability`** dataset —
  reliability labels (and optionally NewsGuard scores) per domain

It normalizes every domain to its registrable form (e.g. `www.cnn.com` →
`cnn.com`) using `tldextract`, then combines the three sets:

- `intersection` mode (default, stricter): domain must be in the general news
  universe **and** labeled reliable, **and not** on the Iffy blocklist
- `union` mode (looser): reliable-labeled domains OR (universe domains minus
  blocklist)

Optional `--min-newsguard-score N` further restricts to only well-scored
domains. Output: `data/trusted_domains.csv`, `data/trusted_domains.json`,
`data/unreliable_domains.json`. (Note: `data/` doesn't exist until you run
this script.)

### `gdelt_client.py` — GDELT API wrapper

One function, `search_articles(query, max_records)`, hits the public
[GDELT Doc API](https://api.gdeltproject.org/api/v2/doc/doc) with
`mode=ArtList`, sorted by `HybridRel` (hybrid relevance), and returns the raw
list of article dicts (title, url, domain, seendate, language, etc.).

### `trust_policy.py` — loads the allowlist at app startup

`load_trusted_domains()` reads `data/trusted_domains.json` (built by the
script above) into a Python `set`. Raises `FileNotFoundError` with a helpful
message if you haven't run the build script yet. Also has `is_trusted_url()`
for one-off checks.

### `verifier.py` — the core scoring logic

- `filter_trusted_articles()` — keeps only GDELT articles whose registrable
  domain is in the trusted set
- `corroboration_score()` — counts the number of **distinct trusted domains**
  reporting on the claim (not article count — this prevents one outlet
  republishing the same story 5 times from inflating the score)
- `verdict_from_score()` — simple rule-based verdict:
  - 3+ independent trusted sources → `SUPPORTED`
  - exactly 2 → `LIKELY_SUPPORTED`
  - 0-1 → `INSUFFICIENT_EVIDENCE`

### `llm_local.py` — local LLM verification layer

Uses `llama-cpp-python` to run a local GGUF model (you supply the model
file — nothing is bundled). Key design choices:

- `build_evidence_pack()` formats up to 8 trusted articles as numbered
  citations: `[1] Title — domain — date — url` — **metadata only, no full
  article text** (keeps prompts small and avoids scraping/paywall issues)
- `build_prompt()` is a strict instruction: verify *only* from the given
  evidence, cite `[#]` after each claim, output one of
  `SUPPORTED / LIKELY_SUPPORTED / NOT_SUPPORTED / INSUFFICIENT_EVIDENCE`
- The LLM only runs if there are ≥2 trusted sources (`min_sources_to_run`)
- If the model's output doesn't contain any `[#]` citation markers, the code
  discards it and substitutes a canned `INSUFFICIENT_EVIDENCE` response — a
  guardrail against uncited/hallucinated verdicts

### `app.py` — FastAPI application (entry point)

Wires everything together:

1. Loads trusted domains once at startup
2. Optionally loads the local LLM if `LLM_MODEL_PATH` env var is set (with
   `LLM_N_CTX`, `LLM_GPU_LAYERS` also configurable) — if loading fails, it's
   recorded but doesn't crash the app
3. `POST /verify` endpoint:
   - Fetches GDELT articles → filters to trusted → computes rule-based
     verdict
   - Returns rule-based results immediately (`gdelt_hits`, `trusted_hits`,
     `unique_trusted_sources`, `trusted_domains`, `verdict_rule_based`, top
     10 trusted articles)
   - Then layers on LLM status: `disabled` (no model configured), `error`
     (model failed to load), `skipped` (fewer than 2 trusted sources), or
     `ran` (full LLM report attached)

### `requirements.txt`

`fastapi`, `uvicorn` (web server), `requests` (HTTP), `pandas`/`pyarrow`
(CSV/dataset handling), `tldextract` (domain parsing), `datasets`
(HuggingFace dataset loading), `llama-cpp-python` (local LLM inference).

### `trustlens-gdelt.zip`

A zipped snapshot/backup of the same project bundled into the repo — not
code you need to read separately.

## Design philosophy worth noting

- **Defense in depth against misinformation**: rather than trusting any
  single source, it requires domain-diversity corroboration (2-3+
  independent trusted outlets) before even considering something verified.
- **LLM is evidence-constrained, not knowledge-based**: it never lets the
  model answer from its own training data — only from retrieved,
  trust-filtered headlines — and it validates the output format (citations)
  before trusting the model's own claim of having cited sources.
- **Everything runs locally except two API calls** (GDELT search, and the
  one-time domain-list downloads) — no article scraping, no cloud LLM calls
  at request time.

## Running it

1. `python build_trusted_domains.py --mode intersection` (needs internet
   access to Zenodo/Iffy/HuggingFace)
2. `uvicorn app:app --reload`

The LLM step is optional — without `LLM_MODEL_PATH` set, you still get the
rule-based verdict.
