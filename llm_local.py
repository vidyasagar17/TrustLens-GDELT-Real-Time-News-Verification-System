from __future__ import annotations
from typing import Dict, List, Optional
import os
import re

from llama_cpp import Llama

def build_evidence_pack(claim: str, trusted_articles: List[Dict], limit: int = 8) -> str:
    lines = []
    lines.append(f"CLAIM: {claim}")
    lines.append("")
    lines.append("EVIDENCE (trusted sources only):")
    for i, a in enumerate(trusted_articles[:limit], 1):
        title = (a.get("title") or "").strip() or "Untitled"
        url = (a.get("url") or "").strip()
        date = (a.get("seendate") or "").strip()
        domain = (a.get("domain") or "").strip()
        lines.append(f"[{i}] {title} — {domain} — {date} — {url}")
    return "\n".join(lines)

def build_prompt(evidence_pack: str) -> str:
    return f"""Verify the claim using ONLY the evidence below. Do not use outside knowledge.

If evidence does not clearly support the claim, say INSUFFICIENT_EVIDENCE.
If evidence contradicts the claim, say NOT_SUPPORTED.
Cite evidence items like [1], [2] after each statement.

Output format:
Verdict: <SUPPORTED|LIKELY_SUPPORTED|NOT_SUPPORTED|INSUFFICIENT_EVIDENCE>
Summary:
- <bullet> [#]
- <bullet> [#]
- <bullet> [#]
Citations: [#], [#], ...

EVIDENCE:
{evidence_pack}
""".strip()

class LocalLlamaVerifier:

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download a GGUF model and set LLM_MODEL_PATH env var, or pass a path."
            )

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count() or 4,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate_report(
        self,
        claim: str,
        trusted_articles: List[Dict],
        min_sources_to_run: int = 2,
        evidence_limit: int = 8,
        max_tokens: int = 350,
        temperature: float = 0.1,
    ) -> Dict:
        evidence_pack = build_evidence_pack(claim, trusted_articles, limit=evidence_limit)

        if len(trusted_articles) < min_sources_to_run:
            return {
                "ran_llm": False,
                "llm_text": "INSUFFICIENT_EVIDENCE: Not enough trusted sources to generate an LLM report.",
                "evidence_pack": evidence_pack,
            }

        prompt = build_prompt(evidence_pack)

        out = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["</s>"],
        )
        text = (out["choices"][0]["text"] or "").strip()

        if not re.search(r"\[\d+\]", text):
            text = (
                "Verdict: INSUFFICIENT_EVIDENCE\n"
                "Summary:\n"
                "- The model did not provide citation-backed statements. [1]\n"
                "Citations: [1]\n"
            )

        return {
            "ran_llm": True,
            "llm_text": text,
            "evidence_pack": evidence_pack,
        }
