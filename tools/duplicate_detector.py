"""
duplicate_detector.py  -  TF-IDF cosine similarity duplicate detection for HOMEBASE.

Checks a candidate title + description against all existing open registry items
before any new item is written. Deterministic — no LLM call involved.

Design
------
- Dual-channel comparison: full text (title+description) AND title-only
  The max of both scores is used — catches paraphrased titles even when
  descriptions differ completely (e.g. "Replace HVAC air filter" vs
  "HVAC Air Filter Replacement" with different descriptions)
- TfidfVectorizer with bigram support fits on the live registry corpus at
  call time (small corpus, negligible latency)
- Configurable similarity threshold (default 0.65)
- Closed items excluded from comparison by default via status_filter

Enterprise analog
-----------------
Deduplication pipeline for intake queues (RMA, ServiceNow, Jira) — prevents
duplicate tickets from being opened for the same underlying issue, a key step
in the VA RMA submitter checklist ("search for a duplicate or similar request").
"""

from typing import TypedDict


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class DuplicateMatch(TypedDict):
    item_id: str
    title: str
    category: str
    status: str
    score: float          # cosine similarity 0.0-1.0 (max of full + title-only)
    score_pct: int        # rounded percentage for display


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.55  # Lower than single-channel — dual approach catches more


def _build_corpus_text(item: dict) -> str:
    """Combine title and description into a single string for vectorization."""
    title = item.get("title", "") or ""
    description = item.get("description", "") or ""
    return f"{title} {description}".strip()


def _vectorize_and_score(candidate_text: str, corpus_texts: list[str]) -> list[float]:
    """
    Fit TF-IDF on corpus+candidate, return cosine similarity scores.
    Returns zeros on failure (e.g. all-stopword vocabulary).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not corpus_texts or not candidate_text.strip():
        return [0.0] * len(corpus_texts)

    all_docs = corpus_texts + [candidate_text]
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(all_docs)
    except ValueError:
        return [0.0] * len(corpus_texts)

    candidate_vec = tfidf_matrix[-1]
    corpus_vecs = tfidf_matrix[:-1]
    return cosine_similarity(candidate_vec, corpus_vecs).flatten().tolist()


def check_duplicates(
    title: str,
    description: str,
    threshold: float = DEFAULT_THRESHOLD,
    status_filter: list[str] | None = None,
    registry: list[dict] | None = None,
) -> list[DuplicateMatch]:
    """
    Check candidate title + description against the existing registry.

    Uses dual-channel scoring:
    - Channel 1: full text (title + description) vs existing full texts
    - Channel 2: title only vs existing titles
    Final score = max(channel1, channel2) per item.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return []

    if status_filter is None:
        status_filter = ["open", "in_progress"]

    if registry is None:
        from tools.registry_tools import get_registry
        registry = get_registry()

    candidates = [
        item for item in registry
        if item.get("status", "open") in status_filter
    ]

    if not candidates:
        return []

    candidate_text = f"{title} {description}".strip()
    if not candidate_text:
        return []

    # Channel 1: full text
    full_corpus = [_build_corpus_text(item) for item in candidates]
    full_scores = _vectorize_and_score(candidate_text, full_corpus)

    # Channel 2: title only
    title_corpus = [(item.get("title", "") or "") for item in candidates]
    title_scores = _vectorize_and_score(title, title_corpus)

    # Max of both channels per item
    combined_scores = [
        max(full_scores[i], title_scores[i])
        for i in range(len(candidates))
    ]

    matches: list[DuplicateMatch] = []
    for i, score in enumerate(combined_scores):
        if score >= threshold:
            item = candidates[i]
            matches.append(DuplicateMatch(
                item_id=item["id"],
                title=item["title"],
                category=item.get("category", "general"),
                status=item.get("status", "open"),
                score=round(float(score), 3),
                score_pct=round(float(score) * 100),
            ))

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def has_duplicates(
    title: str,
    description: str,
    threshold: float = DEFAULT_THRESHOLD,
    registry: list[dict] | None = None,
) -> bool:
    """Return True if any registry item exceeds the similarity threshold."""
    return len(check_duplicates(title, description, threshold=threshold, registry=registry)) > 0


def top_match(
    title: str,
    description: str,
    threshold: float = DEFAULT_THRESHOLD,
    registry: list[dict] | None = None,
) -> DuplicateMatch | None:
    """Return the highest-scoring match, or None if below threshold."""
    matches = check_duplicates(title, description, threshold=threshold, registry=registry)
    return matches[0] if matches else None