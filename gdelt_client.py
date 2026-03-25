import requests
from typing import List, Dict

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

def search_articles(query: str, max_records: int = 50) -> List[Dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
    }
    r = requests.get(GDELT_DOC_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("articles", [])
