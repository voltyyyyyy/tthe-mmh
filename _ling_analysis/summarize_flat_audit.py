import json, collections
rows=[json.loads(l) for l in open(r'_ling_analysis/flat-audit.jsonl',encoding='utf-8') if l.strip()]
print('total audit rows', len(rows))
print('events', collections.Counter(r.get('event') for r in rows))
for ev in rows:
    if ev.get('event')=='staged':
        p=ev.get('proposal',{})
        g=p.get('guideline',{})
        print(f"STAGED round={ev['round']} rule={ev['rule']} patch={ev['patch']} phi={g.get('phi')!r} psi={g.get('psi')!r} judge_conf={p.get('judge_confidence')}")
print('\nVALIDATIONS by rule:')
by=collections.defaultdict(list)
for ev in rows:
    if ev.get('event')=='validation':
        by[ev['rule']].append(ev)
for rule,evs in by.items():
    dec=collections.Counter(e['decision'] for e in evs)
    print(f'\n{rule}: n={len(evs)} decisions={dict(dec)}')
    for e in evs:
        c=e.get('control') or []; t=e.get('treatment') or []
        cp=sum(1 for x in c if x.get('passed')); tp=sum(1 for x in t if x.get('passed'))
        print(f"  r{e['round']:>2} subset={e['subset'][:8]} delta={e['delta']:+.1f} dec={e['decision']:<10} control={cp}/{len(c)} treatment={tp}/{len(t)}")