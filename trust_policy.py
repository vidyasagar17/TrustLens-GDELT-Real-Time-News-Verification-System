import json
from pathlib import Path
from typing import Set
import tldextract

DATA_DIR = Path(__file__).resolve().parent / "data"
TRUSTED_PATH = DATA_DIR / "trusted_domains.json"

def _registrable_domain(url_or_domain: str) -> str:
    s = (url_or_domain or "").strip()
    if not s:
        return ""
    ext = tldextract.extract(s)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()

def load_trusted_domains() -> Set[str]:
    if not TRUSTED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRUSTED_PATH}. Run build_trusted_domains.py first."
        )
    obj = json.loads(TRUSTED_PATH.read_text(encoding="utf-8"))
    return set(obj["trusted_domains"])

def is_trusted_url(url: str, trusted_domains: Set[str]) -> bool:
    d = _registrable_domain(url)
    return d in trusted_domains
