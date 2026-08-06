from typing import Dict, List, Set, Tuple
import tldextract

def registrable_domain(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    ext = tldextract.extract(s)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()

def partition_articles(
    articles: List[Dict], trusted_domains: Set[str], unreliable_domains: Set[str]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split GDELT hits into trusted / flagged-unreliable / unclassified, for cross-checking."""
    trusted, flagged_unreliable, unclassified = [], [], []
    for a in articles:
        url = a.get("url", "") or ""
        dom = a.get("domain", "") or ""
        d = registrable_domain(dom) or registrable_domain(url)
        if d and d in trusted_domains:
            trusted.append(a)
        elif d and d in unreliable_domains:
            flagged_unreliable.append(a)
        else:
            unclassified.append(a)
    return trusted, flagged_unreliable, unclassified

def corroboration_score(trusted_articles: List[Dict]) -> Tuple[int, List[str]]:
    """Coverage score: how many distinct trusted domains mention the topic at all.

    NOTE: this measures topic coverage, not corroboration. A widely-debunked claim
    scores high here because debunkings are themselves trusted-outlet coverage.
    Use stance_corroboration() for an actual support/refute count.
    """
    domains = {registrable_domain(a.get("domain") or a.get("url") or "") for a in trusted_articles}
    domains = {d for d in domains if d}
    return len(domains), sorted(domains)

def verdict_from_score(score: int) -> str:
    """Coverage-only verdict. Kept for comparison against the stance verdict."""
    if score >= 3:
        return "SUPPORTED"
    if score == 2:
        return "LIKELY_SUPPORTED"
    return "INSUFFICIENT_EVIDENCE"


# --- Stance-aware corroboration -------------------------------------------------

SUPPORT_THRESHOLD = 3       # distinct domains for a firm verdict
LIKELY_THRESHOLD = 2        # distinct domains for a hedged verdict
DOMINANCE_RATIO = 3.0       # how far one side must outweigh the other to win outright


def stance_corroboration(articles: List[Dict], stances: List) -> Dict:
    """Count *distinct domains* per stance, so one prolific outlet cannot vote twice."""
    buckets: Dict[str, set] = {"SUPPORTS": set(), "REFUTES": set(), "DISCUSSES": set()}

    for article, stance in zip(articles, stances):
        label = getattr(stance, "label", None) or (stance or {}).get("label")
        if label not in buckets:
            continue  # UNAVAILABLE / unknown -> no vote
        d = registrable_domain(article.get("domain") or article.get("url") or "")
        if d:
            buckets[label].add(d)

    return {
        "support_domains": sorted(buckets["SUPPORTS"]),
        "refute_domains": sorted(buckets["REFUTES"]),
        "discuss_domains": sorted(buckets["DISCUSSES"]),
        "support_count": len(buckets["SUPPORTS"]),
        "refute_count": len(buckets["REFUTES"]),
        "discuss_count": len(buckets["DISCUSSES"]),
    }


def verdict_from_stance(support_count: int, refute_count: int) -> str:
    """Verdict from directional evidence.

    Unlike verdict_from_score, this can express REFUTED and DISPUTED -- outcomes
    the coverage-only pipeline is structurally unable to represent.
    """
    s, r = support_count, refute_count

    if s == 0 and r == 0:
        return "INSUFFICIENT_EVIDENCE"

    # Both sides materially represented -> the claim is contested, not settled.
    if s >= LIKELY_THRESHOLD and r >= LIKELY_THRESHOLD:
        return "DISPUTED"

    if s > r:
        if r > 0 and s < r * DOMINANCE_RATIO:
            return "DISPUTED"
        if s >= SUPPORT_THRESHOLD:
            return "SUPPORTED"
        if s >= LIKELY_THRESHOLD:
            return "LIKELY_SUPPORTED"
        return "INSUFFICIENT_EVIDENCE"

    if r > s:
        if s > 0 and r < s * DOMINANCE_RATIO:
            return "DISPUTED"
        if r >= SUPPORT_THRESHOLD:
            return "REFUTED"
        if r >= LIKELY_THRESHOLD:
            return "LIKELY_REFUTED"
        return "INSUFFICIENT_EVIDENCE"

    return "DISPUTED"  # s == r and both non-zero


def confidence_signals(coverage_score: int, stance: Dict) -> Dict:
    """Break out the inputs behind a verdict so the caller can see *why*."""
    s = stance["support_count"]
    r = stance["refute_count"]
    directional = s + r
    return {
        "unique_trusted_domains": coverage_score,
        "directional_sources": directional,
        "support_ratio": round(s / directional, 3) if directional else None,
        # What fraction of covering outlets took any position at all. Low values
        # mean the verdict rests on a thin slice of the retrieved evidence.
        "directional_coverage": round(directional / coverage_score, 3) if coverage_score else None,
    }
