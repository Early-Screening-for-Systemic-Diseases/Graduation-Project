# nlpv2/pipeline/fusion.py
# Stage 5 — Normalization + Gate 1 + Entropy Fusion
#
# CRITICAL RULES (from spec):
#   - Always normalize before fusion — scores are on different scales.
#   - Branches that fail Gate 1 (max score < 0.50) are muted, not removed.
#   - Entropy measures how "spread out" a branch's scores are.
#     A branch that strongly favors one disease has LOW entropy → HIGH trust.
#   - Do NOT skip normalization before fusion.

import math

# ── Constants ────────────────────────────────────────────────────────────────
BRANCH_THRESHOLD = 0.50   # Gate 1: branch must have at least one score >= this
MUTED_WEIGHT     = 0.05   # Trust assigned to a branch that fails Gate 1


# ── Step 1 — Min-Max Normalization ───────────────────────────────────────────

def normalize_scores(scores: dict) -> dict:
    """
    Min-max normalize a dict of {disease: score} to [0, 1].

    If all scores are equal (hi == lo), the branch has no discriminative
    opinion — return 0.5 for all diseases (neutral / no signal).

    Args:
        scores: {disease_name: float}

    Returns:
        Normalized {disease_name: float} in [0, 1].
    """
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        # Flat distribution — branch has no opinion
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# ── Step 2 — Gate 1 ──────────────────────────────────────────────────────────

def branch_passes_gate(scores: dict) -> bool:
    """
    Gate 1: A branch passes only if its maximum score >= BRANCH_THRESHOLD.
    A branch that fails is muted (trust = MUTED_WEIGHT), not removed.

    Args:
        scores: Normalized {disease_name: float} scores.

    Returns:
        True if the branch has a meaningful signal, False otherwise.
    """
    return max(scores.values()) >= BRANCH_THRESHOLD


# ── Step 3 — Entropy and Trust ───────────────────────────────────────────────

def shannon_entropy(scores: dict) -> float:
    """
    Compute Shannon entropy over the score distribution.

    Low entropy  → branch strongly favors one disease → HIGH trust.
    High entropy → branch is uncertain / spread out   → LOW trust.

    Args:
        scores: Normalized {disease_name: float} probabilities.

    Returns:
        Entropy value H >= 0.
    """
    return -sum(
        p * math.log(p + 1e-9)
        for p in scores.values()
        if p > 0
    )


def entropy_to_trust(H: float) -> float:
    """
    Convert entropy to a trust weight in (0, 1].

    trust(H) = 1 / (1 + H)

    H = 0 (perfect certainty) → trust = 1.0
    H → ∞ (maximum uncertainty) → trust → 0.0

    Args:
        H: Shannon entropy value.

    Returns:
        Trust weight in (0, 1].
    """
    return 1.0 / (1.0 + H)


# ── Step 4 — Full Fusion Function ────────────────────────────────────────────

def entropy_fusion(
    lex_results:  dict,
    bio_results:  dict,
    bart_results: dict,
    diseases:     list,
) -> dict:
    """
    Stage 5 — Normalize, apply Gate 1, compute entropy-weighted fusion.

    Pipeline:
        1. Extract raw scores from each branch.
        2. Normalize each branch independently (min-max).
        3. Apply Gate 1: mute branches with no signal.
        4. Compute Shannon entropy per branch → convert to trust weight.
        5. Normalize trust weights so they sum to 1.
        6. Compute weighted average fused score per disease.

    BART temperature scaling is applied in Stage 4 (bart_branch.py).
    This function receives the calibrated_score from BART.

    Args:
        lex_results:  Output of lexicon_branch() — contains 'idf_score'.
        bio_results:  Output of biobert_branch() — contains 'similarity'.
        bart_results: Output of bart_branch()    — contains 'calibrated_score'.
        diseases:     List of disease names.

    Returns:
        {
          'Diabetes': {
            'fused_score': float,
            'weights': {'lexicon': float, 'biobert': float, 'bart': float},
            'component_scores': {'lexicon': float, 'biobert': float, 'bart': float}
          },
          ...
        }
    """
    # ── Extract raw scores ───────────────────────────────────────────────────
    lex_raw  = {d: lex_results[d]["idf_score"]         for d in diseases}
    bio_raw  = {d: bio_results[d]["similarity"]         for d in diseases}
    bart_raw = {d: bart_results[d]["calibrated_score"]  for d in diseases}

    # ── Normalize each branch independently ─────────────────────────────────
    lex_scores  = normalize_scores(lex_raw)
    bio_scores  = normalize_scores(bio_raw)
    bart_scores = normalize_scores(bart_raw)

    # ── Gate 1 + entropy trust ───────────────────────────────────────────────
    if branch_passes_gate(lex_scores):
        lex_trust = entropy_to_trust(shannon_entropy(lex_scores))
    else:
        lex_trust = MUTED_WEIGHT

    if branch_passes_gate(bio_scores):
        bio_trust = entropy_to_trust(shannon_entropy(bio_scores))
    else:
        bio_trust = MUTED_WEIGHT

    if branch_passes_gate(bart_scores):
        bart_trust = entropy_to_trust(shannon_entropy(bart_scores))
    else:
        bart_trust = MUTED_WEIGHT

    # ── Normalize trust weights to sum to 1 ──────────────────────────────────
    total = lex_trust + bio_trust + bart_trust
    w_lex, w_bio, w_bart = lex_trust / total, bio_trust / total, bart_trust / total

    # ── Weighted average per disease ─────────────────────────────────────────
    fused = {}
    for d in diseases:
        score = (
            w_lex  * lex_scores[d] +
            w_bio  * bio_scores[d] +
            w_bart * bart_scores[d]
        )
        fused[d] = {
            "fused_score": score,
            "weights": {
                "lexicon": round(w_lex,  4),
                "biobert": round(w_bio,  4),
                "bart":    round(w_bart, 4),
            },
            "component_scores": {
                "lexicon": round(lex_scores[d],  4),
                "biobert": round(bio_scores[d],  4),
                "bart":    round(bart_scores[d], 4),
            },
        }

    return fused
