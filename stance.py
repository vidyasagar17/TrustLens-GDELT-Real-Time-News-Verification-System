"""
Stance detection: does an article SUPPORT, REFUTE, or merely DISCUSS a claim?

This is the component that separates "reputable outlets are covering these words"
from "reputable outlets assert this is true". Without it, corroboration_score
measures topic coverage, and widely-debunked claims score highest because they
attract the most trusted-outlet coverage.

Implemented with an NLI (natural language inference) model:
    premise    = article text (title, or title + body if available)
    hypothesis = the claim
    entailment    -> SUPPORTS
    contradiction -> REFUTES
    neutral       -> DISCUSSES

The model is loaded lazily and the module degrades gracefully if `transformers`
is not installed, so the rest of the pipeline still runs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging
import os

log = logging.getLogger(__name__)

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
DISCUSSES = "DISCUSSES"
UNAVAILABLE = "UNAVAILABLE"

DEFAULT_MODEL = os.environ.get("STANCE_MODEL", "microsoft/deberta-v3-base-mnli")

# A bare headline is a weak premise -- it often omits the negation or attribution
# that decides stance ("Officials deny X" vs "X confirmed"). We demand more
# confidence before acting on a title-only judgement than on a full-body one.
TITLE_ONLY_MIN_CONFIDENCE = 0.80
FULL_TEXT_MIN_CONFIDENCE = 0.60

# Rough char count above which we assume we have body text, not just a headline.
BODY_TEXT_THRESHOLD = 300


@dataclass
class StanceResult:
    label: str
    confidence: float
    premise_chars: int
    title_only: bool
    abstained: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _article_premise(article: Dict) -> str:
    """Best available text for an article. Uses body if a fetcher populated it."""
    parts = [
        (article.get("title") or "").strip(),
        (article.get("body") or article.get("text") or "").strip(),
    ]
    return ". ".join(p for p in parts if p)


class StanceClassifier:
    """NLI-backed stance classifier. Model loads on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: int = -1,
        max_premise_chars: int = 2000,
    ):
        self.model_name = model_name
        self.device = device
        self.max_premise_chars = max_premise_chars
        self._pipe = None
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._pipe is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._pipe is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                top_k=None,  # return scores for all three NLI classes
                truncation=True,
            )
            log.info("Loaded stance model: %s", self.model_name)
        except Exception as e:  # missing dep, no network, OOM, ...
            self._load_error = f"{type(e).__name__}: {e}"
            log.warning("Stance model unavailable (%s)", self._load_error)

    @staticmethod
    def _map_nli_label(raw: str) -> str:
        r = (raw or "").strip().upper()
        if r.startswith("ENTAIL") or r == "LABEL_2":
            return SUPPORTS
        if r.startswith("CONTRADICT") or r == "LABEL_0":
            return REFUTES
        return DISCUSSES

    def classify(self, claim: str, article: Dict) -> StanceResult:
        self._ensure_loaded()

        premise = _article_premise(article)[: self.max_premise_chars]
        title_only = len(premise) < BODY_TEXT_THRESHOLD

        if self._pipe is None:
            return StanceResult(
                label=UNAVAILABLE,
                confidence=0.0,
                premise_chars=len(premise),
                title_only=title_only,
                abstained=True,
                reason=f"stance_model_unavailable: {self._load_error}",
            )

        if not premise:
            return StanceResult(
                label=DISCUSSES,
                confidence=0.0,
                premise_chars=0,
                title_only=True,
                abstained=True,
                reason="empty_premise",
            )

        try:
            scores = self._pipe({"text": premise, "text_pair": claim})
        except Exception as e:
            return StanceResult(
                label=UNAVAILABLE,
                confidence=0.0,
                premise_chars=len(premise),
                title_only=title_only,
                abstained=True,
                reason=f"inference_error: {type(e).__name__}",
            )

        if scores and isinstance(scores[0], list):  # pipeline may nest per-input
            scores = scores[0]

        best = max(scores, key=lambda d: d["score"])
        label = self._map_nli_label(best["label"])
        confidence = float(best["score"])

        floor = TITLE_ONLY_MIN_CONFIDENCE if title_only else FULL_TEXT_MIN_CONFIDENCE
        if label in (SUPPORTS, REFUTES) and confidence < floor:
            # Not confident enough to count this as a directional vote.
            return StanceResult(
                label=DISCUSSES,
                confidence=confidence,
                premise_chars=len(premise),
                title_only=title_only,
                abstained=True,
                reason=f"below_confidence_floor({floor})",
            )

        return StanceResult(
            label=label,
            confidence=confidence,
            premise_chars=len(premise),
            title_only=title_only,
        )

    def classify_all(self, claim: str, articles: List[Dict]) -> List[StanceResult]:
        return [self.classify(claim, a) for a in articles]
