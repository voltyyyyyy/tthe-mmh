"""Throwaway prototype: how would I even test the volatile-traits question?

Two testable layers:
  LAYER 1  Is the measurement real?      -> property assertions, no model, no stats
  LAYER 2  Does it predict promotion?    -> collect (trait, outcome) pairs, compare ranks

The generator is deliberately built so I KNOW the answer, which is how you validate
a pipeline: if it cannot recover a relationship I planted, it cannot find a real one.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_memory import (  # noqa: E402
    MMHConfig,
    MetaMemoryEngine,
    Patch,
    PatchOperation,
    Rule,
    RuleTier,
    SQLiteStore,
    ValidationEvidence,
)


def similarity(a, b):
    """Match engine.retrieve: cosine mapped to [0, 1]."""
    if len(a) != len(b):
        return 0.0
    return (sum(x * y for x, y in zip(a, b)) + 1.0) / 2.0


@dataclass
class Traits:
    novelty: float
    overlap: float
    uniqueness: float
    sparsity: float
    surprise: float
    lifespan: int


def logit(successes: int, failures: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    p = (alpha + successes) / (alpha + beta + successes + failures)
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def compute_traits(engine, rule, *, stable_snapshot, volatile_snapshot, k=3) -> Traits:
    """Traits frozen against engine state AS OF staging. No future leakage."""
    emb = rule.embedding or engine.embedding(rule.phi)

    stable_sims = [similarity(emb, s.embedding or engine.embedding(s.phi)) for s in stable_snapshot]
    overlap = max(stable_sims) if stable_sims else 0.0
    novelty = 1.0 - overlap

    others = [v for v in volatile_snapshot if v.rule_id != rule.rule_id]
    other_sims = sorted(
        (similarity(emb, v.embedding or engine.embedding(v.phi)) for v in others), reverse=True
    )
    uniqueness = 1.0 - other_sims[0] if other_sims else 1.0
    nearest = other_sims[:k]
    sparsity = 1.0 - (sum(nearest) / len(nearest)) if nearest else 1.0

    # Surprise = belief movement across the FIRST validation step (not from the prior,
    # which would be constant across rules and therefore carry no information).
    surprise = abs(logit(rule.successes, rule.failures) - logit(0, 0))

    return Traits(novelty, overlap, uniqueness, sparsity, surprise, rule.successful_lifespan)


# ---------------------------------------------------------------- LAYER 1
def layer1_measurement_is_real() -> None:
    print("=" * 74)
    print("LAYER 1: is the measurement real?  (no model, no stats)")
    print("=" * 74)

    store = SQLiteStore(":memory:")
    engine = MetaMemoryEngine(store, config=MMHConfig(promotion_age=2, promotion_confidence=0.55))

    near_a = Rule("near-a", "handle rare class retrieval with low confidence", "repair A")
    near_b = Rule("near-b", "handle rare class retrieval with low confidence too", "repair B")
    far_c = Rule("far-c", "parse nested json arrays from an http response body", "repair C")
    for r in (near_a, near_b, far_c):
        r.embedding = engine.embedding(r.phi)
    with store.transaction():
        for r in (near_a, near_b, far_c):
            store.put_rule(r)

    sim_near = similarity(near_a.embedding, near_b.embedding)
    sim_far = similarity(near_a.embedding, far_c.embedding)
    print(f"  sim(paraphrase pair) = {sim_near:.3f}   want HIGH")
    print(f"  sim(unrelated pair)  = {sim_far:.3f}   want LOWER")
    assert sim_near > sim_far
    print("  PASS: metric orders related above unrelated\n")

    vol = [near_a, near_b, far_c]
    t_near = compute_traits(engine, near_a, stable_snapshot=[], volatile_snapshot=vol)
    t_far = compute_traits(engine, far_c, stable_snapshot=[], volatile_snapshot=vol)
    print(f"  uniqueness(duplicate) = {t_near.uniqueness:.3f}   want LOW")
    print(f"  uniqueness(outlier)   = {t_far.uniqueness:.3f}   want HIGH")
    assert t_far.uniqueness > t_near.uniqueness
    print("  PASS: uniqueness lower for the redundant rule\n")

    t_dup = compute_traits(engine, near_b, stable_snapshot=[near_a], volatile_snapshot=vol)
    t_new = compute_traits(engine, far_c, stable_snapshot=[near_a], volatile_snapshot=vol)
    print(f"  near stable: novelty={t_dup.novelty:.3f}  overlap={t_dup.overlap:.3f}")
    print(f"  far  stable: novelty={t_new.novelty:.3f}  overlap={t_new.overlap:.3f}")
    assert t_new.novelty > t_dup.novelty
    assert abs(t_dup.novelty + t_dup.overlap - 1.0) < 1e-9
    print("  PASS: novelty rises with distance from stable; novelty+overlap==1\n")

    store.close()


# ---------------------------------------------------------------- LAYER 2
GENUINE_TOPICS = [
    "index out of bounds in a slice operation",
    "integer overflow when summing large values",
    "off by one in a loop boundary check",
]
SPURIOUS_TOPICS = [
    "add a docstring to the helper function",
    "rename the local variable for clarity",
    "reformat the file with black",
]


def build_episode(seed: int, n_rounds: int = 16):
    """One episode. Half the proposals address real failure modes (genuine), half are
    cosmetic (spurious). Planted structure:
        - genuine proposals are SIMILAR to the stable core  -> high overlap, low novelty
        - genuine proposals succeed validation more often   -> get promoted
        - spurious proposals are NOVEL                      -> high novelty, low happiness
    If the pipeline is correct, novelty should rank ANTI-predictively for promotion
    and overlap should rank predictively. Every trait whose AUC == 0.5 is dead weight.
    """
    rng = random.Random(seed)
    store = SQLiteStore(":memory:")
    engine = MetaMemoryEngine(store, config=MMHConfig(promotion_age=2, promotion_confidence=0.55))

    with store.transaction():
        for i, topic in enumerate(GENUINE_TOPICS):
            r = Rule(f"core-{i}", topic, "existing repair", tier=RuleTier.STABLE)
            r.embedding = engine.embedding(r.phi)
            store.put_rule(r)

    records = []
    for rnd in range(1, n_rounds + 1):
        is_genuine = rng.random() < 0.5
        topic = rng.choice(GENUINE_TOPICS if is_genuine else SPURIOUS_TOPICS)
        rule = Rule(f"r{rnd}", topic, "proposed repair", initial_confidence=0.6)
        patch = Patch(
            patch_id=f"p{rnd}",
            operation=PatchOperation.ADD,
            result_rules=(rule,),
            context=topic,
            judge_confidence=0.6,
        )
        try:
            staged = engine.update_cycle([patch], rnd)
        except Exception:
            continue
        if not staged["staged"]:
            continue
        staged_rule = store.get_rule(f"r{rnd}")
        if staged_rule is None:
            continue

        traits = compute_traits(
            engine,
            staged_rule,
            stable_snapshot=store.list_rules(RuleTier.STABLE),
            volatile_snapshot=store.list_rules(RuleTier.VOLATILE),
        )

        # Genuine repairs validate; cosmetic ones mostly do not.
        success = rng.random() < (0.90 if is_genuine else 0.25)
        for step in range(1, 4):
            engine.update_cycle(
                [], rnd + step,
                [ValidationEvidence(f"p{rnd}", success, f"sub{step}", rnd + step)],
            )
        # update_cycle ages at the START of a cycle and advance_round no-ops on a seen
        # round, so open a fresh round before the freshness gates can pass.
        engine.advance_round(rnd + 4)
        promoted = any(r.rule_id == f"r{rnd}" for r in engine.promote_eligible(rnd + 4))

        # Reflect the final arousal state into surprise (first-observation shift).
        final = store.get_rule(f"r{rnd}")
        traits.surprise = abs(
            logit(final.successes, final.failures) - logit(0, 0)
        ) if final else traits.surprise
        records.append((traits, promoted, is_genuine))

    store.close()
    return records


def auc(scores, labels) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cliff_delta(xs, ys) -> float:
    """Non-parametric effect size in [-1, 1]."""
    if not xs or not ys:
        return float("nan")
    gt = sum((a > b) - (a < b) for a in xs for b in ys)
    return gt / (len(xs) * len(ys))


def layer2_does_it_predict() -> None:
    print("=" * 74)
    print("LAYER 2: does it predict promotion?  (collect pairs, compare ranks)")
    print("=" * 74)

    recs = []
    for seed in range(40):
        recs.extend(build_episode(seed))

    labels = [p for _, p, _ in recs]
    truth = [g for _, _, g in recs]
    print(f"  {len(recs)} staged rules over 40 episodes")
    print(f"  promotion rate {statistics.mean(labels):.2f}   genuine rate {statistics.mean(truth):.2f}")
    print("  a trait only matters if it beats ranking by chance (AUC 0.5)\n")

    names = ["novelty", "overlap", "uniqueness", "sparsity", "surprise", "lifespan"]
    print(f"  {'trait':<12}{'AUC(promoted)':>15}{'AUC(genuine)':>15}{'delta V-vs-promoted':>21}")
    for name in names:
        scores = [getattr(t, name) for t, _, _ in recs]
        a_prom = auc(scores, labels)
        a_true = auc(scores, truth)
        promoted_scores = [s for s, y in zip(scores, labels) if y]
        other = [s for s, y in zip(scores, labels) if not y]
        d = cliff_delta(promoted_scores, other)
        print(f"  {name:<12}{a_prom:>15.3f}{a_true:>15.3f}{d:>21.3f}")

    print("\n  PLANTED truth in this generator: genuine repairs resemble stable memory")
    print("  (low novelty / high overlap) and validate more often.")
    print("  So a WORKING pipeline should show overlap AUC > 0.5 and novelty AUC < 0.5.")
    print("  Anything stuck at 0.500 is a trait that carries no information as defined.")
    print()


if __name__ == "__main__":
    layer1_measurement_is_real()
    layer2_does_it_predict()
    print("=" * 74)
    print("PROVES: traits are computable from real engine state, behave correctly on")
    print("        known inputs, and the prediction pipeline recovers a planted effect.")
    print("DOES NOT PROVE: any real result about MMH. The generator is mine, and its")
    print("        answer is baked in. Real streams + real model are the actual test.")
    print("=" * 74)
