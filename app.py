from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import os

from trust_policy import load_trusted_domains, load_unreliable_domains
from pipeline import run_verification
from stance import StanceClassifier
from llm_local import LocalLlamaVerifier

app = FastAPI(title="TrustLens-GDELT")

TRUSTED = load_trusted_domains()
UNRELIABLE = load_unreliable_domains()

LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "").strip()
LLM_N_CTX = int(os.environ.get("LLM_N_CTX", "4096"))
LLM_GPU_LAYERS = int(os.environ.get("LLM_GPU_LAYERS", "0"))
STANCE_ENABLED = os.environ.get("STANCE_ENABLED", "1") != "0"

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

STANCE = StanceClassifier() if STANCE_ENABLED else None


class VerifyRequest(BaseModel):
    claim: str
    max_records: int = 50


@app.get("/health")
def health():
    return {
        "trusted_domains": len(TRUSTED),
        "unreliable_domains": len(UNRELIABLE),
        "stance_model": (
            "ready" if (STANCE and STANCE.available) else "unavailable"
        ),
        "stance_error": STANCE.load_error if STANCE else None,
        "llm": "ready" if LLM else ("error" if LLM_LOAD_ERROR else "disabled"),
        "llm_error": LLM_LOAD_ERROR,
    }


@app.post("/verify")
async def verify(req: VerifyRequest):
    # Both llama.cpp inference and NLI are synchronous and CPU-bound; running
    # them inline would block the event loop for every concurrent request.
    result = await run_in_threadpool(
        run_verification,
        claim=req.claim,
        trusted_domains=TRUSTED,
        unreliable_domains=UNRELIABLE,
        max_records=req.max_records,
        stance_classifier=STANCE,
        llm=LLM,
    )
    if LLM_LOAD_ERROR:
        result["llm_status"] = "error"
        result["llm_error"] = LLM_LOAD_ERROR
    return result
