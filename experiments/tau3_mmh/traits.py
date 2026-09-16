"""Volatile traits: how volatile memory differs from stable/long-term memory.

Every trait is computed against the engine state **as of the round the rule was staged**,
using the embedding provider configured on the engine.  Freezing matters: ``novelty`` and
``overlap`` are distances to the *stable* set, so if they were recomputed later a rule's
own promotion would retroactively redefine its novelty.  A trait measured post-promotion
would partly measure the outcome it is supposed to predict.

Lifespan is reported but is a **gate variable**, not a finding.  The promotion gate
requires ``successful_lifespan >= promotion_age`` and ``elapsed_age >= promotion_age``, so
"volatile rules are younger than stable rules" restates ``engine.py`` rather than
discovering anything.  It is included as a control the other traits must beat.

Provider choice changes these numbers materially.  MMH's default embedding is normalized
feature hashing, which scores ``cos(paraphrase) = 0.0`` on a verified pair -- it measures
token overlap, not meaning.  Use a semantic provider (Qwen3-Embedding) for these traits.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from meta_memory import Rule, RuleTier


@dataclass(frozen=True)
class TraitVector:
    """One rule's trait values, frozen at staging."""

    rule_id: str
    staged_round: int
    tier_at_measure: str
    novelty: float
    overlap: float
    uniqueness: float
    sparsity: float
    surprise: float
    lifespan: int
    elapsed_age: int
    n_stable: int
    n_volatile: int
    is_stable_now: str = ""          # filled by the analyser, not at measure time
    promoted: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Match ``MetaMemoryEngine.retrieve``: cosine mapped to [0, 1]."""
    return (_cosine(a, b) + 1.0) / 2.0


def _logit(successes: int, failures: int, alpha: float, beta: float) -> float:
    p = (alpha + successes) / (alpha + beta + successes + failures)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def compute_traits(
    engine: Any,
    rule: Rule,
    *,
    stable_snapshot: Sequence[Rule],
    volatile_snapshot: Sequence[Rule],
    k: int = 3,
) -> TraitVector:
    """All traits against the supplied snapshots. Callers must freeze those at staging.

    ``k`` is the neighbourhood size for ``sparsity``.  A rule can simultaneously be the
    *unique* copy of a *crowded* region, which is why uniqueness (a minimum) and sparsity
    (a local mean) are reported separately rather than collapsed.
    """
    embedding = rule.embedding or engine.embedding(rule.phi)

    stable_sims = [
        _similarity(embedding, s.embedding or engine.embedding(s.phi)) for s in stable_snapshot
    ]
    overlap = max(stable_sims) if stable_sims else 0.0
    novelty = 1.0 - overlap

    others = [v for v in volatile_snapshot if v.rule_id != rule.rule_id]
    other_sims = sorted(
        (_similarity(embedding, v.embedding or engine.embedding(v.phi)) for v in others),
        reverse=True,
    )
    uniqueness = 1.0 - other_sims[0] if other_sims else 1.0
    nearest = other_sims[:k]
    sparsity = 1.0 - (sum(nearest) / len(nearest)) if nearest else 1.0

    prior = rule.provenance.get("confidence_prior") if isinstance(rule.provenance, Mapping) else None
    alpha = float(prior.get("alpha", 1.0)) if isinstance(prior, Mapping) else 1.0
    beta = float(prior.get("beta", 1.0)) if isinstance(prior, Mapping) else 1.0
    # Surprise as |log-odds shift| from the weak prior to the observed posterior.
    # NOTE: if computed from a validation outcome this partly ENCODES the outcome and
    # must not be presented as a predictor of it. See ex_ante_surprise().
    surprise = abs(_logit(rule.successes, rule.failures, alpha, beta) - _logit(0, 0, alpha, beta))

    return TraitVector(
        rule_id=rule.rule_id,
        staged_round=int(rule.created_round or 0),
        tier_at_measure=rule.tier.value,
        novelty=novelty,
        overlap=overlap,
        uniqueness=uniqueness,
        sparsity=sparsity,
        surprise=surprise,
        lifespan=int(rule.successful_lifespan),
        elapsed_age=int(rule.elapsed_age),
        n_stable=len(stable_snapshot),
        n_volatile=len(volatile_snapshot),
    )


def ex_ante_surprise(
    engine: Any,
    context: str,
    precedents: Sequence[Any],
    *,
    bandwidth: float = 1.0,
) -> float:
    """Surprise computed BEFORE any outcome exists: how unusual is this context.

    Uses the precedent log's empirical outcome base rate weighted by context similarity.
    This is the version safe to use as a *predictor*; the posterior-shift version is
    partly a restatement of the outcome.  Returns 0.0 when there is no usable precedent.
    """
    if not precedents:
        return 0.0
    embedding = engine.embedding(context)
    weights: list[float] = []
    outcomes: list[float] = []
    for precedent in precedents:
        vec = tuple(precedent.context_embedding)
        if len(vec) != len(embedding):
            continue
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding, vec)))
        weight = math.exp(-(distance * distance) / (2 * bandwidth * bandwidth))
        weights.append(weight)
        outcomes.append(float(precedent.outcome))
    if not weights or sum(weights) <= 0.0:
        return 0.0
    base_rate = sum(w * o for w, o in zip(weights, outcomes)) / sum(weights)
    base_rate = min(max(base_rate, 1e-3), 1.0 - 1e-3)
    # High surprise = context historically unlike a success.
    return -math.log(base_rate)


def summarise_traits(rows: Sequence[TraitVector]) -> dict[str, Any]:
    """Per-trait central tendency for a group (e.g. all volatile rules)."""
    if not rows:
        return {}
    names = ("novelty", "overlap", "uniqueness", "sparsity", "surprise", "lifespan")
    out: dict[str, Any] = {"n": len(rows)}
    for name in names:
        values = [getattr(r, name) for r in rows]
        out[name] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return out


def cliffs_delta(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1].

    Preferred over a p-value here: at these sample sizes significance is easy and
    meaningless, while the magnitude of separation is the actual claim.
    """
    if not xs or not ys:
        return float("nan")
    gt = sum((a > b) - (a < b) for a in xs for b in ys)
    return gt / (len(xs) * len(ys))


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank AUC with tie handling (Mann-Whitney form). 0.5 means no signal."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation; used to detect redundant traits before they are combined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")

    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return float("nan") if denominator == 0 else numerator / denominator
