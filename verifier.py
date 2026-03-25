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

def filter_trusted_articles(articles: List[Dict], trusted_domains: Set[str]) -> List[Dict]:
    out = []
    for a in articles:
        url = a.get("url", "") or ""
        dom = a.get("domain", "") or ""
        d = registrable_domain(dom) or registrable_domain(url)
        if d and d in trusted_domains:
            out.append(a)
    return out

def corroboration_score(trusted_articles: List[Dict]) -> Tuple[int, List[str]]:
    domains = {registrable_domain(a.get("domain") or a.get("url") or "") for a in trusted_articles}
    domains = {d for d in domains if d}
    return len(domains), sorted(domains)

def verdict_from_score(score: int) -> str:
    if score >= 3:
        return "SUPPORTED"
    if score == 2:
        return "LIKELY_SUPPORTED"
    return "INSUFFICIENT_EVIDENCE"
