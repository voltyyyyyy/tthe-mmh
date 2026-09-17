"""Paired task outcomes and mechanism diagnostics; no claim from missing data."""
import argparse
import json
import math
from pathlib import Path
import random
import sqlite3

from .full_study import ARMS, write_json
from .traits import cliffs_delta, auc


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def paired(left, right):
    a, b = {r['task']:r['passed'] for r in left}, {r['task']:r['passed'] for r in right}
    if set(a) != set(b) or not a:
        return {'available':False,'reason':'unmatched or missing tasks'}
    wins = sum(a[t] and not b[t] for t in a)
    losses = sum(b[t] and not a[t] for t in a)
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant,k) for k in range(min(wins,losses)+1)) / 2**discordant) if discordant else 1.
    delta = [int(a[t])-int(b[t]) for t in sorted(a)]
    rng = random.Random(42)
    samples = sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(2000))
    return {'available':True,'n':len(a),'wins':wins,'losses':losses,
            'difference':sum(delta)/len(delta),'bootstrap_95ci':[samples[50],samples[1949]],
            'mcnemar_exact_p':p,'unit':'task; not independent study seeds'}


def analyze(folder):
    results, report = {}, {}
    for arm in ARMS:
        rounds = sorted((json.loads(p.read_text()) for p in (folder/arm).glob('round-*.json')),
                        key=lambda x:x['round'])
        results[arm] = {'search':[r for rd in rounds for r in rd['records']]}
        for split in ('validation','test'):
            results[arm][split] = [r for p in (folder/arm/'evaluations').glob(f'final-{split}-*/result.json')
                                  for r in json.loads(p.read_text())]
        audit = read_jsonl(folder/arm/'audit.jsonl')
        # Interrupted round replays append audit records; canonical event identity removes duplicates.
        audit = list({(e['event'],e['round'],e.get('rule',e.get('patch',''))):e for e in audit}.values())
        report[arm] = {'scores':{s:{'n':len(rows),'passed':sum(r['passed'] for r in rows)}
                                 for s,rows in results[arm].items()},
                       'staged':sum(e['event']=='staged' for e in audit),
                       'promoted':sum(len(e.get('rules',[])) for e in audit if e['event']=='promotion'),
                       'validation_decisions':{decision:sum(e.get('decision')==decision for e in audit)
                                               for decision in ('success','failure','indecisive')},
                       'stable_probes':[e for e in audit if e.get('tier')=='stable'],
                       'rounds':[{'round':r['round'],'domain':r['domain'],
                                  'n':len(r['records']),'passed':sum(t['passed'] for t in r['records']),
                                  'injected':r['injected'],'counts':r['counts']} for r in rounds]}
        traits = list({r['rule_id']:r for r in read_jsonl(folder/arm/'traits.jsonl')}.values())
        promotions = {rid for e in audit if e['event']=='promotion' for rid in e.get('rules',[])}
        for row in traits:
            row['promoted'] = int(row['rule_id'] in promotions)
        trait_report = {'n':len(traits),'promoted':len(promotions),
                        'status':'descriptive only; gate-controlled predictive analysis requires adequate events',
                        'note':'novelty = 1 - overlap; no stable reference when n_stable = 0', 'traits':{}}
        if len(promotions) and len(promotions) < len(traits):
            for trait in ('novelty','overlap','uniqueness','sparsity','ex_ante_surprise'):
                yes = [r[trait] for r in traits if r['promoted']]
                no = [r[trait] for r in traits if not r['promoted']]
                trait_report['traits'][trait] = {'cliffs_delta':cliffs_delta(yes,no)}
        report[arm]['trait_analysis'] = trait_report
        write_json(folder/arm/'traits-final.json',traits)
    comparisons = {split:{'mmh_vs_flat':paired(results['mmh'][split],results['flat'][split]),
                          'mmh_vs_baseline':paired(results['mmh'][split],results['baseline'][split])}
                   for split in ('search','validation','test')}
    complete = all(report[a]['scores']['search']['n']==225 and
                   report[a]['scores']['validation']['n']==75 and
                   report[a]['scores']['test']['n']==75 for a in ARMS)
    final = {'complete':complete,'arms':report,'paired':comparisons,
             'research_conclusion':'Not established: inspect mechanism counts, paired uncertainty and replicate streams.',
             'limitations':['one trial/seed; no pass^2 or pass^3 estimate',
                            'domain difficulty shifts alone do not prove concept drift',
                            'no stable promotion means stable-vs-volatile trait question is unidentifiable',
                            'unpromoted rules include right-censored late proposals']}
    write_json(folder/'analysis.json',final)
    lines = ['# All375 study status',f'Complete coverage: {complete}',
             '', '| Arm | Search | Validation (reused) | Final test | Promoted |',
             '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        scores = report[arm]['scores']
        line = [f"{scores[s]['passed']}/{scores[s]['n']}" for s in ('search','validation','test')]
        lines.append('| '+arm+' | '+' | '.join(line)+f" | {report[arm]['promoted']} |")
    lines += ['', 'This is one exploratory stream per model. Domain shifts alone do not establish',
              'concept drift. No inference about the tiers is warranted unless the mechanism',
              'activated and the stable probes and paired comparisons support it.',
              '', 'See analysis.json for task pairing, uncertainty, trait diagnostics and limitations.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.out)
