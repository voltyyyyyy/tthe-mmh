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

## Paper conflicts and chosen resolution

The draft is internally inconsistent in a few places.  No implementation can
obey every sentence simultaneously, so this code follows **Algorithm 1 and the
numbered equations** when an appendix example disagrees:

1. **Eq. 5 vs Appendix B lifecycle numbers.** Appendix B shows confidence
   `0.50 -> 0.78 -> 0.84 -> 0.89` after three successes; Eq. 5 with the shipped
   `alpha = beta = 1` prior gives `0.50 -> 0.667 -> 0.75 -> 0.80`. The engine
   follows Eq. 5.
2. **Eq. 11 vs Appendix B influence example.** Appendix B multiplies raw
   similarities (`0.89 * 1 + 0.78 * 1 + 0.45 * 0`) directly; Eq. 11 specifies
   Gaussian kernel weights. The engine follows Eq. 11 and sums over the full
   precedent log as written.
3. **Stable edit gate vs Appendix B Round 10.** Algorithm 1 says stable
   `MERGE`/`DELETE` requires repeated cross-subset underperformance. Appendix B
   merges two stable rules because their `phi`s overlap, with no failure
   history. The engine follows Algorithm 1; a stable redundancy merge still
   needs the failure gate.
4. **Merged-rule placement.** Section 3.2 and Algorithm 1 say a new edit is
   first written to the volatile tier as pending. Appendix B Round 10 shows the
   merged stable rule as stable/pending. The engine follows Section 3.2 and
   stages merged results in volatile until promotion.
5. **Age/lifespan wording.** The prose calls `a_t` the number of rounds a rule
   has remained volatile, while Appendix Prompt 1 defines `lifespan` as
   successful validation rounds. The engine tracks both (`elapsed_age` and
   `successful_lifespan`) and requires both to reach the promotion-age threshold.

These choices are intentional and covered by the conformance tests.


## Choices made where the draft is underspecified

- Default prior: `alpha = beta = 1`; influence threshold: `0.6`; promotion
  confidence: `0.8`; promotion age: `3`; context-match threshold: `0.7`.
  `MMHConfig.alpha`/`beta` are honored and written into each engine-staged rule.
- `lifespan` counts successful validation rounds. Calendar/round age is tracked
  separately so a patch cannot be promoted simply by waiting; promotion requires
  both `successful_lifespan` and `elapsed_age` to reach the age threshold.
- Initial judge confidence and posterior confidence are distinct. Appendix
  Prompt 1's `new_rule.confidence` is accepted as the cold-start judge score when
  no separate `judge_confidence` field is present; validation recomputes the
  posterior.
- Promotion requires at least two distinct validation subsets and recovery of
  the recorded parent failure when a failure case was recorded. These are
  conservative operationalisations of the prose requirement for repeated,
  cross-subset evidence and recovery.
- A stable rule accepts only DELETE or MERGE proposals after five distinct
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

## Embedding engines

`MetaMemoryEngine` accepts any callable `text -> vector`.  The default remains
the dependency-free deterministic hash embedding so offline tests and demos run
without credentials.

For semantic similarity, use the packaged `EmbeddingEngine`:

```python
from meta_memory import EmbeddingConfig, EmbeddingEngine, MetaMemoryEngine

embedding = EmbeddingEngine(EmbeddingConfig(
    provider="openai",
    model="text-embedding-3-small",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
))
engine = MetaMemoryEngine(store, embedding_provider=embedding)
```

The LiveCodeBench MMH integration also builds the provider from environment
variables, so no code change is needed:

```text
MMH_EMBEDDING_PROVIDER=openai
MMH_EMBEDDING_MODEL=text-embedding-3-small
MMH_EMBEDDING_BASE_URL=https://api.openai.com/v1
MMH_EMBEDDING_API_KEY_ENV=OPENAI_API_KEY
MMH_EMBEDDING_DIMENSIONS=1536        # optional
MMH_EMBEDDING_TIMEOUT=60
MMH_EMBEDDING_BATCH_SIZE=64
```

Local open-source models are supported when the optional
`sentence-transformers` package is installed:

```text
MMH_EMBEDDING_PROVIDER=sentence-transformers
MMH_EMBEDDING_LOCAL_MODEL=BAAI/bge-m3
MMH_EMBEDDING_DEVICE=cuda            # optional; cpu/default otherwise
MMH_EMBEDDING_TRUST_REMOTE_CODE=0    # set 1 only for models requiring it
```

`BAAI/bge-m3` is the default local model: 1024-dimensional, multilingual, and
strong on general retrieval contexts.  For a smaller/faster prototype use
`BAAI/bge-small-en-v1.5`; for code-heavy retrieval use
`jinaai/jina-embeddings-v2-base-code`.

`EmbeddingEngine` caches per-text vectors, batches API calls, and normalizes
vectors before returning them.


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
