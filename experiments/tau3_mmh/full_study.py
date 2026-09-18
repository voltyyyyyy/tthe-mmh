"""Resumable, paired all375 study. Search alone supplies proposer trajectories.

Each domain has six maturation rounds. Validation groups are disjoint and their
identities follow their task sets, never the round number. Final test is evaluated
only after all adaptation ends. Flat memory uses the same initial candidate
validation but has no promotion delay or tier-specific ordering.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from meta_memory import MMHConfig, SQLiteStore, build_embedding_engine, PatchStatus, RuleTier
from .guidelines import GuidelineMemory
from .proposer import FailureSignature, LLMProposer
from .tau3_binding import EvalInvocation, load_per_task, classify_error, write_guidelines
from .traits import compute_traits, ex_ante_surprise

DOMAINS = ('airline', 'retail', 'telecom', 'banking_knowledge')
ARMS = ('baseline', 'mmh', 'flat')
CANDIDATE_EVALS_PER_ROUND = 2
TRANSPORT = ('ReadTimeout', 'ConnectTimeout', 'ConnectError', 'APIConnectionError',
             'NotFoundError', 'InternalServerError', 'ServiceUnavailable',
             'CUDA out of memory', 'RemoteProtocolError', 'LocalProtocolError',
             'Connection refused', 'Connection reset', 'HTTPStatusError')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, default=str, allow_nan=False) + '\n')
        f.flush()


def task_key(task):
    return task.replace(':', '_', 1)


def partition(items, n):
    return [list(items[i * len(items) // n:(i + 1) * len(items) // n]) for i in range(n)]


def candidate_rules(memory, round_id, max_maturing=CANDIDATE_EVALS_PER_ROUND):
    """Choose validation candidates without starving older rules.

    Pending patches must each be resolved.  Validated volatile rules then get up
    to ``max_maturing`` additional probes, oldest evidence first.  The previous
    ``candidates[:2]`` policy let each newly staged rule consume one slot forever,
    so older rules never accumulated ``successful_lifespan``.
    """
    rules = memory.store.list_rules()
    pending = [r for r in rules
               if r.status is PatchStatus.PENDING and (r.created_round or 0) < round_id]
    maturing = [r for r in rules
                if r.status is not PatchStatus.PENDING
                and r.tier is RuleTier.VOLATILE
                and (r.created_round or 0) < round_id]
    pending.sort(key=lambda r: (r.provenance.get('last_measured', r.created_round or 0),
                                r.created_round or 0, r.rule_id))
    maturing.sort(key=lambda r: (r.provenance.get('last_measured', r.created_round or 0),
                                 r.created_round or 0, r.rule_id))
    return pending + maturing[:max_maturing]


def make_schedule(splits):
    """All search tasks once; six rounds/domain, with three validation groups."""
    schedule = []
    for domain in DOMAINS:
        tasks = [f'{domain}:{t}' for t in splits['search'][domain]]
        groups = partition([f'{domain}:{t}' for t in splits['validation'][domain]], 3)
        for i, batch in enumerate(partition(tasks, 6)):
            schedule.append({'round': len(schedule) + 1, 'domain': domain,
                             'tasks': batch, 'validation': groups[i % 3],
                             'subset': digest(sorted(groups[i % 3]))})
    return schedule


def inspect_record(record, trace_dir):
    """Only structured error events can reclassify a scored model failure.

    Do not search natural-language conversations for error words. Recovered model
    retries are retained as telemetry, not automatically excluded from scoring.
    """
    error = classify_error(record)
    events = []
    path = Path(trace_dir) / 'events.jsonl'
    if path.exists():
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    terminal = [e for e in events if e.get('type') == 'agent_error']
    for event in terminal:
        detail = str(event.get('error', ''))
        if any(marker in detail for marker in TRANSPORT):
            error = detail
        # Harness bugs and exhausted model budgets remain failures, by design.
    if not events or not any(e.get('type') == 'grading' for e in events):
        error = error or 'missing official grading trace'
    for event in events:
        if event.get('error') and event.get('type') in {'observation', 'customer_error'}:
            detail = str(event.get('result', event.get('error')))
            if any(marker in detail for marker in TRANSPORT):
                error = detail
    return error, events


class Study:
    candidate_evals_per_round = CANDIDATE_EVALS_PER_ROUND

    def __init__(self, repo, out, concurrency=2, model='Qwen/Qwen3.8-27B',
                 base_url='http://127.0.0.1:8100/v1'):
        self.repo, self.out = Path(repo).resolve(), Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.concurrency = concurrency
        self.benchmark = self.repo / 'reproduction/qwen_tau3/all375/benchmark.yaml'
        self.split_path = self.benchmark.with_name('split_manifest.json')
        self.splits = json.loads(self.split_path.read_text())['splits']
        sets = {s: {f'{d}:{t}' for d, ids in ds.items() for t in ids}
                for s, ds in self.splits.items()}
        assert len(set.union(*sets.values())) == 375
        assert sum(map(len, sets.values())) == 375
        self.schedule = make_schedule(self.splits)
        self.model = model
        self.url = base_url
        self.harness = Path(__file__).parent / 'tau3_harness'
        self.embedding = build_embedding_engine()
        if self.embedding.config.provider == 'hash':
            raise RuntimeError('Real study requires the pinned semantic embedding provider')
        probe = self.embedding.embed_batch(['Retrieve the booking before changing a flight',
                                            'Look up the reservation before modifying an itinerary',
                                            'Bake bread in a hot oven'])
        self.embedding_probe = {'dimensions': len(probe[0]),
                                'paraphrase_cosine': sum(a*b for a,b in zip(probe[0],probe[1])),
                                'unrelated_cosine': sum(a*b for a,b in zip(probe[0],probe[2]))}
        if len(probe[0]) != 1024 or self.embedding_probe['paraphrase_cosine'] <= self.embedding_probe['unrelated_cosine']:
            raise RuntimeError('Embedding semantic sanity check failed')
        self.memories = {}
        for arm in ARMS:
            folder = self.out / arm
            folder.mkdir(exist_ok=True)
            self.memories[arm] = GuidelineMemory(store=SQLiteStore(folder / 'memory.sqlite'),
                                               embedding_provider=self.embedding, config=MMHConfig())
        self.proposer = LLMProposer(base_url=self.url, model=self.model, max_failures_shown=2)
        self.candidate_evals_per_round = CANDIDATE_EVALS_PER_ROUND
        self.freeze_manifest()

    def freeze_manifest(self):
        code = list(Path(__file__).parent.glob('*.py')) + list((self.repo / 'meta_memory').glob('*.py'))
        sources = {str(p.relative_to(self.repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in code + [self.benchmark, self.split_path, self.harness / 'harness.py',
                                    self.repo / 'benchmarks/tau3/adapter.py',
                                    self.repo / 'benchmarks/tau3/program_adapter.py']}
        manifest = {'protocol': 'all375-paired-v1', 'model': self.model, 'endpoint': self.url,
                    'sources': sources, 'embedding': asdict(self.embedding.config),
                    'embedding_probe': self.embedding_probe, 'schedule': self.schedule,
                    'arms': ARMS, 'max_prompt_rules': 8, 'candidate_evals_per_round': self.candidate_evals_per_round,
                    'candidate_selection': 'all pending + oldest maturing up to candidate_evals_per_round',
                    'mmh_config': asdict(MMHConfig()), 'concurrency': self.concurrency,
                    'validation_decision': 'paired reward sum improvement; ties indecisive',
                    'flat_definition': 'initial paired validation retained; validated rules one pool; no promotion gate',
                    'limitations': ['one search stream/seed; exploratory, not a replicated efficacy claim',
                                    'domain shifts need not imply conditional concept drift',
                                    'validation data repeatedly consulted; not final test',
                                    'promotion counts may be too small for trait prediction'],
                    'test_trials': 1}
        target = self.out / 'manifest.json'
        if target.exists():
            old = json.loads(target.read_text())
            expected = json.loads(json.dumps(manifest, default=str))
            # Recomputed floating-point sanity probes are not protocol identifiers.
            old.pop('embedding_probe', None)
            expected.pop('embedding_probe', None)
            if old != expected:
                raise RuntimeError('Protocol/source changed: refuse mixed-provenance resume')
        else:
            write_json(target, manifest)
            archive = self.out / 'source'
            for p in code + [self.benchmark, self.split_path, self.harness / 'harness.py']:
                dest = archive / p.relative_to(self.repo)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
            for name, cmd in [('git.txt', ['git', 'rev-parse', 'HEAD']),
                              ('diff.patch', ['git', 'diff']),
                              ('gpu.txt', ['nvidia-smi']),
                              ('processes.txt', ['ps', '-u', 'yangfan', '-o', 'pid,ppid,lstart,args'])]:
                result = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True)
                (archive / name).write_text(result.stdout + result.stderr)

    def status(self, **kwargs):
        data = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs}
        write_json(self.out / 'status.json', data)
        print(json.dumps(data), flush=True)

    def active(self, arm):
        rules = self.memories[arm].active()
        if arm == 'flat':
            rules.sort(key=lambda r: (-(r.created_round or 0), r.rule_id))
        return rules[:8]

    def evaluate(self, arm, label, tasks, split, rules):
        signature = {'arm': arm, 'label': label, 'tasks': tasks, 'split': split,
                     'rules': [(r.rule_id, r.phi, r.psi) for r in rules]}
        folder = self.out / arm / 'evaluations' / (label + '-' + digest(signature)[:12])
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / 'result.json'
        if result_path.exists():
            return json.loads(result_path.read_text())
        write_json(folder / 'request.json', signature)
        prompt_file = folder / 'guidelines.md'
        write_guidelines(rules, prompt_file)
        completed = {}
        partial = folder / 'partial.json'
        if partial.exists():
            completed = json.loads(partial.read_text())
        for attempt in range(1, 4):
            pending = [t for t in tasks if task_key(t) not in completed]
            if not pending:
                break
            name = 'study_' + digest(str(self.out))[:8] + '_' + uuid.uuid4().hex[:14]
            invocation = EvalInvocation(repo=self.repo, benchmark=str(self.benchmark) + ':' + split,
                                        config_dir=self.harness, name=name, tasks=pending,
                                        model=self.model, concurrency=self.concurrency,
                                        python=sys.executable, timeout_s=86400)
            env = {**os.environ, 'PYTHONPATH': str(self.repo), 'OPENAI_API_KEY': 'EMPTY',
                   'LOCAL_MODEL_BASE_URL': self.url, 'LOCAL_OPENAI_V1_BASE': self.url,
                   'OPENAI_BASE_URL': self.url, 'OPENAI_API_BASE': self.url,
                   'MMH_GUIDELINES_FILE': str(prompt_file), 'PYTHONUNBUFFERED': '1',
                   'TAU3_TASK_TIMEOUT_S': '3600', 'META_AGENT_CONCURRENCY': str(self.concurrency),
                   'TAU2_DATA_DIR': '/home/yangfan/meta-agent-test/tau2-bench/data'}
            # Ambient Azure routing must not silently divert the frozen user simulator.
            for key in ('AZURE_OPENAI_V1_BASE','AZURE_FOUNDRY_OPENAI_BASE','AZURE_API_BASE',
                        'AZURE_FOUNDRY_API_KEY','AZURE_API_KEY'):
                env.pop(key, None)
            self.status(state='running', arm=arm, evaluation=label, pending=len(pending),
                        candidate=name, attempt=attempt)
            log = folder / (name + '.log')
            write_json(folder / (name + '.command.json'), invocation.command())
            with log.open('w') as stream:
                process = subprocess.run(invocation.command(), cwd=self.repo, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, timeout=86400)
            paths = list((self.repo / 'experience').glob(f'*/candidates/{name}/scores.json'))
            if len(paths) != 1:
                append_json(self.out / 'infrastructure.jsonl', {'name':name, 'returncode': process.returncode,
                                                             'error':'missing or ambiguous scores'})
                continue
            score_path = paths[0]
            records = load_per_task(score_path)
            keys = [r.get('task_name') or r.get('task') for r in records]
            if len(keys) != len(set(keys)) or set(keys) != {task_key(t) for t in pending}:
                raise RuntimeError(f'Task coverage mismatch for {name}')
            for record, key in zip(records, keys):
                trace = Path(record.get('trial_dir') or '/nonexistent')
                error, events = inspect_record(record, trace)
                archive = folder / 'attempts' / name / digest(key)[:16]
                archive.mkdir(parents=True, exist_ok=True)
                write_json(archive / 'record.json', record)
                if trace.is_dir():
                    shutil.copytree(trace, archive / 'traces', dirs_exist_ok=True)
                if error:
                    append_json(self.out / 'infrastructure.jsonl', {'name':name, 'task':key, 'error':error,
                                                                 'archive':str(archive)})
                    continue
                reward = record.get('reward')
                if reward is None or not math.isfinite(float(reward)):
                    raise RuntimeError('Missing/nonfinite official reward')
                completed[key] = {'task': key, 'reward': float(reward), 'passed': float(reward) > 0,
                                  'turns':record.get('num_turns'), 'wall_time_s':record.get('wall_time_s'),
                                  'trace_dir':str(archive / 'traces'), 'candidate':name,
                                  'model_errors':sum(e.get('type') == 'model_error' for e in events)}
            write_json(partial, completed)
        if len(completed) != len(tasks):
            raise RuntimeError(f'Unresolved infrastructure errors in {folder}; valid tasks preserved for resume')
        result = [completed[task_key(t)] for t in tasks]
        write_json(result_path, result)
        return result

    def failure_evidence(self, records, domain):
        failures = []
        for record in records:
            if record['passed']:
                continue
            path = Path(record['trace_dir']) / 'tau2_conversation.jsonl'
            visible = path.read_text() if path.exists() else ''
            # This file contains the agent/user trajectory only, not grader assertions.
            failures.append(FailureSignature(domain=domain, task_id=record['task'],
                            signature='official_task_failure',
                            evidence={'reward':record['reward'], 'turns':record['turns'],
                                      'conversation':visible[-11000:]}))
        return failures

    def adapt(self, arm, spec, records):
        memory = self.memories[arm]
        round_id = spec['round']
        audit = self.out / arm / 'audit.jsonl'
        # Resolve every pending patch, then give the oldest validated volatile
        # rules additional probes up to candidate_evals_per_round.  This prevents
        # new proposals from starving existing rules of promotion evidence.
        candidates = candidate_rules(memory, round_id, self.candidate_evals_per_round)
        for candidate in candidates:
            active = self.active(arm)
            without = [r for r in active if r.rule_id != candidate.rule_id][:7]
            with_rule = without + [candidate]
            control = self.evaluate(arm, f'r{round_id}-{candidate.rule_id}-control',
                                    spec['validation'], 'validation', without)
            treatment = self.evaluate(arm, f'r{round_id}-{candidate.rule_id}-candidate',
                                      spec['validation'], 'validation', with_rule)
            delta = sum(r['reward'] for r in treatment) - sum(r['reward'] for r in control)
            append_json(audit, {'event':'validation', 'round':round_id, 'rule':candidate.rule_id,
                               'tier':candidate.tier.value, 'subset':spec['subset'], 'delta':delta,
                               'control':control, 'treatment':treatment,
                               'decision':'success' if delta > 0 else 'failure' if delta < 0 else 'indecisive'})
            candidate.provenance['last_measured'] = round_id
            with memory.store.transaction():
                memory.store.put_rule(candidate)
            if delta == 0:
                continue
            if candidate.tier is RuleTier.STABLE:
                memory.record_stable_observation(candidate.rule_id, delta > 0, spec['subset'], round_id)
            else:
                memory.record_round(round_id=round_id, subset_id=spec['subset'],
                                    successes=[candidate.rule_id] if delta > 0 else [],
                                    failures=[candidate.rule_id] if delta < 0 else [])
        if arm == 'mmh':
            promoted = memory.promote(round_id)
            append_json(audit, {'event':'promotion', 'round':round_id,
                               'rules':[r.rule_id for r in promoted]})
        # Flat is intentionally a single validated pool: no forced promotion/tier labels.
        failures = self.failure_evidence(records, spec['domain'])
        proposal_path = self.out / arm / f'proposals-r{round_id}.json'
        if proposal_path.exists():
            from .proposer import Proposal
            from .guidelines import Guideline
            proposals = []
            for p in json.loads(proposal_path.read_text()):
                p['guideline'] = Guideline(**p['guideline'])
                proposals.append(Proposal(**p))
        else:
            proposals = self.proposer.propose(round_id=round_id, failures=failures, memory=memory)
            write_json(proposal_path, [asdict(p) for p in proposals])
            for rejection in getattr(self.proposer, 'last_rejections', []):
                append_json(audit, {'event': 'proposal_rejected', 'round': round_id, **rejection})
        for proposal in proposals[:1]:
            if memory.store.get_patch(f'add:{round_id}:{proposal.guideline.guideline_id}'):
                continue
            before_stable, before_volatile = memory.stable(), memory.volatile()
            from meta_memory import Patch, PatchOperation
            from .guidelines import guideline_to_rule
            patch = Patch(patch_id=f'add:{round_id}:{proposal.guideline.guideline_id}',
                          operation=PatchOperation.ADD,
                          result_rules=(guideline_to_rule(proposal.guideline),),
                          context=proposal.guideline.phi,
                          judge_confidence=proposal.judge_confidence,
                          provenance=proposal.as_provenance())
            eligible = memory.engine.filter_and_resolve([patch])
            if not eligible:
                append_json(audit, {'event':'influence_rejected', 'round':round_id,
                                   'patch':patch.patch_id, 'reason':patch.last_error})
                continue
            patch = memory.stage_patch(eligible[0], round_id)
            rule = memory.store.get_rule(proposal.guideline.guideline_id)
            trait = compute_traits(memory.engine, rule, stable_snapshot=before_stable,
                                   volatile_snapshot=before_volatile).as_dict()
            trait['ex_ante_surprise'] = ex_ante_surprise(memory.engine, rule.phi,
                                                       memory.store.list_precedents())
            append_json(self.out / arm / 'traits.jsonl', trait)
            append_json(audit, {'event':'staged', 'round':round_id, 'patch':patch.patch_id,
                               'rule':rule.rule_id, 'proposal':asdict(proposal)})

    def run(self, smoke=False):
        try:
            if smoke:
                # One search task/domain; never touch final test in plumbing probes.
                tasks = [domain + ':' + str(self.splits['search'][domain][0]) for domain in DOMAINS]
                self.evaluate('baseline', 'smoke-four-domains', tasks, 'search', [])
                self.status(state='smoke_complete')
                return
            for spec in self.schedule:
                round_id = spec['round']
                # Rotate arm ordering to reduce serving-time confounding.
                order = ARMS[round_id % 3:] + ARMS[:round_id % 3]
                for arm in order:
                    marker = self.out / arm / f'round-{round_id}.json'
                    if marker.exists():
                        continue
                    memory = self.memories[arm]
                    start_snapshot = self.out / arm / f'round-{round_id}-start.sqlite'
                    if start_snapshot.exists():
                        # Restore an interrupted round before replaying cached evaluations.
                        import sqlite3
                        source = sqlite3.connect(start_snapshot)
                        source.backup(memory.store.connection)
                        source.close()
                    else:
                        memory.snapshot_to(start_snapshot)
                    memory.advance(round_id)
                    active = self.active(arm) if arm != 'baseline' else []
                    records = self.evaluate(arm, f'search-r{round_id}', spec['tasks'], 'search', active)
                    if arm != 'baseline':
                        self.adapt(arm, spec, records)
                    memory.snapshot_to(self.out / arm / 'memory.sqlite')
                    write_json(marker, {'round':round_id, 'domain':spec['domain'], 'records':records,
                                        'injected':[r.rule_id for r in active],
                                        'gates':memory.promotion_report(), 'counts':memory.tier_counts()})
            # Frozen final memory. Validation coverage is reported separately from test.
            for split in ('validation', 'test'):
                for arm in ARMS:
                    active = self.active(arm) if arm != 'baseline' else []
                    for domain in DOMAINS:
                        tasks = [f'{domain}:{t}' for t in self.splits[split][domain]]
                        self.evaluate(arm, f'final-{split}-{domain}', tasks, split, active)
            self.summarize()
            self.status(state='complete', unique_tasks_per_arm=375)
        except Exception as exc:
            self.status(state='stopped', error=f'{type(exc).__name__}: {exc}')
            raise

    def summarize(self):
        summary = {}
        for arm in ARMS:
            rows = []
            for p in (self.out / arm).glob('round-*.json'):
                rows.extend(json.loads(p.read_text())['records'])
            data = {'search': {'n':len(rows), 'passed':sum(r['passed'] for r in rows)}}
            for split in ('validation', 'test'):
                records = []
                for p in (self.out / arm / 'evaluations').glob(f'final-{split}-*/result.json'):
                    records.extend(json.loads(p.read_text()))
                data[split] = {'n':len(records), 'passed':sum(r['passed'] for r in records)}
            data['tiers'] = self.memories[arm].tier_counts()
            summary[arm] = data
        write_json(self.out / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--base-url', default='http://127.0.0.1:8100/v1')
    args = parser.parse_args()
    Study(args.repo, args.out, args.concurrency, args.model, args.base_url).run(smoke=args.smoke)


if __name__ == '__main__':
    main()
