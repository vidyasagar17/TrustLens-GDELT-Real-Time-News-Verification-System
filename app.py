from fastapi import FastAPI
from pydantic import BaseModel
import os

from gdelt_client import search_articles
from trust_policy import load_trusted_domains
from verifier import filter_trusted_articles, corroboration_score, verdict_from_score
from llm_local import LocalLlamaVerifier

app = FastAPI(title="TrustLens-GDELT")

TRUSTED = load_trusted_domains()

LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "").strip()
LLM_N_CTX = int(os.environ.get("LLM_N_CTX", "4096"))
LLM_GPU_LAYERS = int(os.environ.get("LLM_GPU_LAYERS", "0"))

LLM = None
LLM_LOAD_ERROR = None
if LLM_MODEL_PATH:
    try:
        LLM = LocalLlamaVerifier(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_gpu_layers=LLM_GPU_LAYERS,
        )
    except Exception as e:
        LLM_LOAD_ERROR = str(e)

class VerifyRequest(BaseModel):
    claim: str
    max_records: int = 50

@app.post("/verify")
def verify(req: VerifyRequest):
    raw = search_articles(req.claim, max_records=req.max_records)
    trusted_hits = filter_trusted_articles(raw, TRUSTED)

    score, domains = corroboration_score(trusted_hits)
    verdict_rule = verdict_from_score(score)

    response = {
        "claim": req.claim,
        "gdelt_hits": len(raw),
        "trusted_hits": len(trusted_hits),
        "unique_trusted_sources": score,
        "trusted_domains": domains,
        "verdict_rule_based": verdict_rule,
        "top_trusted_articles": [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "domain": a.get("domain"),
                "seendate": a.get("seendate"),
                "language": a.get("language"),
            }
            for a in trusted_hits[:10]
        ],
    }

    if LLM_LOAD_ERROR:
        response["llm_status"] = "error"
        response["llm_error"] = LLM_LOAD_ERROR
        response["llm_report"] = None
        return response

    if LLM is None:
        response["llm_status"] = "disabled"
        response["llm_report"] = None
        response["llm_hint"] = "Set env var LLM_MODEL_PATH to a local GGUF model path to enable LLM summaries."
        return response

    if score < 2:
        response["llm_status"] = "skipped"
        response["llm_report"] = "INSUFFICIENT_EVIDENCE: Less than 2 independent trusted sources. LLM not invoked."
        return response

    llm_out = LLM.generate_report(
        claim=req.claim,
        trusted_articles=trusted_hits,
        min_sources_to_run=2,
        evidence_limit=8,
        max_tokens=350,
        temperature=0.1,
    )
    response["llm_status"] = "ran" if llm_out["ran_llm"] else "skipped"
    response["llm_report"] = llm_out["llm_text"]
    response["llm_evidence_pack"] = llm_out["evidence_pack"]

    return response
