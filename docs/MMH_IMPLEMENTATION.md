# Meta-Memory Harness (MMH) implementation notes

This repository implements the paper draft's Meta-Memory Harness as a small,
offline-capable library. The implementation is deliberately separate from the
benchmark integrations: a rule is a typed, persisted causal claim and a patch
is a separately persisted, reversible attempted change.

The source draft is vendored unchanged at `docs/papers/mmh-draft.pdf`.

## Paper-to-code mapping

| Draft element | Implementation |
| --- | --- |
| Eq. 1, rule `r = <phi, psi, omega, c, tau>` | `meta_memory.types.Rule` retains these public fields, with IDs, tier, provenance, evidence counters, and timestamps required for safe execution. |
| Eq. 3, stable/volatile union | `RuleTier` and the SQLite-backed store maintain disjoint tiers. |
| Eq. 5, confidence posterior | `MetaMemoryEngine` recomputes `c = (alpha + successes) / (alpha + beta + successes + failures)` from independent validation evidence. |
| Eq. 6--8, promotion | Promotion is an atomic database transition after confidence, age, recovery, and evidence gates pass. |
| Eq. 9--10, atomic operations | `PatchOperation` and `Patch` represent ADD, DELETE, REFINE, SPLIT, and MERGE individually, with parent snapshots for rollback. |
| Eq. 11, influence | `MetaMemoryEngine` calculates the Gaussian similarity-weighted outcome of precedent patches and uses the judge score as a cold-start fallback. |
| Eq. 12, patch status | Pending patches resolve once to validated or rolled back after decisive evidence, then append a precedent. |
| Algorithm 1 | `update_cycle`: propose/reflection, influence/conflict filter, volatile staging and held-out validation, then calibration/promotion. |
| Appendix prompts 1--5 | `meta_memory.adapters` validates the patch, arbitration, context-match, promotion-summary, and fallback-score response shapes. |

## Choices made where the draft is underspecified

- Default prior: `alpha = beta = 1`; influence threshold: `0.6`; promotion
  confidence: `0.8`; promotion age: `3`; context-match threshold: `0.7`.
- `lifespan` counts successful validation rounds. Calendar/round age is tracked
  separately so a patch cannot be promoted simply by waiting.
- Initial judge confidence and posterior confidence are distinct. The former is
  only an influence cold-start signal; validation recomputes the latter.
- Promotion requires at least two distinct validation subsets and recovery of
  the recorded parent failure when a failure case was recorded. These are
  conservative operationalisations of the prose requirement for repeated,
  cross-subset evidence and recovery.
- A stable rule accepts only DELETE or MERGE proposals after five consecutive
  applicable failure rounds spanning at least two subsets. A stable source is
  never overwritten in place; a merged replacement enters as pending and must
  earn promotion.
- The paper does not define a temporal-decay equation. Metadata is preserved;
  repeated observed failures, not elapsed wall time, is used for drift action.
- Precedent fallback is used when fewer than three precedents are within the
  configured Gaussian bandwidth. The shipped offline embedding is deterministic
  normalized feature hashing, not a semantic model.
- The paper's Eq. 8 conflates edits and rules. The implementation applies edits
  transactionally before calculating the next rule sets.

## Standalone versus TTHE

The standalone demo is **labeled**: it uses expected outcomes to make the
paper lifecycle observable in an offline unit-sized stream. This is a test
fixture, not an online benchmark protocol.

TTHE integration is **label-free during adaptation**. Its MMH adapter may use
only public task material and execution evidence produced by the frozen,
public-test executor. Private tests, hidden scores, and grading artifacts are
never put in proposals, rule state, prompts, patch records, or traces. Hidden
evaluation happens after adaptation and is reported separately from proxy
validation.

## Running the offline demonstration

```powershell
python -m meta_memory.demo
```

The output is deterministic JSON. It includes an accepted patch, recurrence on
distinct subsets, promotion to stable memory, a harmful patch rollback, and a
subsequent distribution-shift failure signal. No model credentials, datasets,
or network access are required.
