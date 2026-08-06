"""
Core verification pipeline, independent of the HTTP layer.

app.py and eval/evaluate.py both call run_verification() so the API and the
evaluation harness can never drift apart -- a benchmark that measures different
code than production ships is worse than no benchmark.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from gdelt_client import search_articles
from verifier import (
    partition_articles,
    corroboration_score,
    verdict_from_score,
    stance_corroboration,
    verdict_from_stance,
    confidence_signals,
)


def article_summary(a: Dict) -> Dict:
    return {
        "title": a.get("title"),
        "url": a.get("url"),
        "domain": a.get("domain"),
        "seendate": a.get("seendate"),
        "language": a.get("language"),
    }


def run_verification(
    claim: str,
    trusted_domains: Set[str],
    unreliable_domains: Set[str],
    max_records: int = 50,
    stance_classifier=None,
    llm=None,
    search_fn=search_articles,
    top_n: int = 10,
) -> Dict:
    """Retrieve, partition, score, and (optionally) summarize.

    stance_classifier and llm are optional; the pipeline degrades to
    coverage-only scoring when they are absent.
    """
    raw = search_fn(claim, max_records=max_records)
    trusted, flagged, unclassified = partition_articles(raw, trusted_domains, unreliable_domains)

    score, domains = corroboration_score(trusted)

    result: Dict = {
        "claim": claim,
        "gdelt_hits": len(raw),
        "trusted_hits": len(trusted),
        "unique_trusted_sources": score,
        "trusted_domains": domains,
        "verdict_coverage_based": verdict_from_score(score),
        "top_trusted_articles": [article_summary(a) for a in trusted[:top_n]],
        "flagged_unreliable_hits": len(flagged),
        "top_flagged_unreliable_articles": [article_summary(a) for a in flagged[:top_n]],
        "unclassified_hits": len(unclassified),
        "top_unclassified_articles": [article_summary(a) for a in unclassified[:top_n]],
    }

    # --- stance ---------------------------------------------------------------
    if stance_classifier is None or not stance_classifier.available:
        result["stance_status"] = "disabled"
        result["verdict"] = result["verdict_coverage_based"]
        result["verdict_basis"] = "coverage_only"
        if stance_classifier is not None:
            result["stance_error"] = stance_classifier.load_error
        supporting: List[Dict] = trusted
    else:
        stances = stance_classifier.classify_all(claim, trusted)
        agg = stance_corroboration(trusted, stances)

        result["stance_status"] = "ran"
        result["stance"] = agg
        result["verdict"] = verdict_from_stance(agg["support_count"], agg["refute_count"])
        result["verdict_basis"] = "stance"
        result["signals"] = confidence_signals(score, agg)
        result["article_stances"] = [
            {**article_summary(a), **s.to_dict()} for a, s in zip(trusted[:top_n], stances[:top_n])
        ]
        # Only articles that actually support the claim are eligible as evidence.
        supporting = [a for a, s in zip(trusted, stances) if s.label == "SUPPORTS"]

    # --- LLM summary ----------------------------------------------------------
    if llm is None:
        result["llm_status"] = "disabled"
        result["llm_report"] = None
        return result

    if result["verdict"] in ("INSUFFICIENT_EVIDENCE",):
        result["llm_status"] = "skipped"
        result["llm_report"] = "INSUFFICIENT_EVIDENCE: not enough directional evidence; LLM not invoked."
        return result

    # The LLM renders the verdict the rule engine already chose -- it never decides.
    evidence = supporting if supporting else trusted
    llm_out = llm.generate_report(
        claim=claim,
        trusted_articles=evidence,
        fixed_verdict=result["verdict"],
    )
    result["llm_status"] = "ran" if llm_out.get("ran_llm") else "skipped"
    result["llm_report"] = llm_out.get("llm_text")
    result["llm_evidence_pack"] = llm_out.get("evidence_pack")
    return result
