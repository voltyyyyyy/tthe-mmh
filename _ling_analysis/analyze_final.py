import json, glob, os, itertools, math, random, collections
base=r'_ling_analysis/final'
arms=['baseline','mmh','flat']
data={a:{} for a in arms}
for a in arms:
    for p in glob.glob(os.path.join(base,a,'evaluations','final-*','result.json')):
        parts=os.path.basename(os.path.dirname(p)).split('-')
        split=parts[1]; domain='-'.join(parts[2:-1])
        rows=json.load(open(p,encoding='utf-8'))
        data[a].setdefault(split,{})[domain]={r['task']:r for r in rows}
print('TASK COUNTS')
for split in ['validation','test']:
    for a in arms:
        n=sum(len(d) for d in data[a][split].values()); p=sum(r['passed'] for d in data[a][split].values() for r in d.values())
        print(split,a,n,p)

def mcnemar(wins,losses):
    n=wins+losses
    if n==0: return 1.0
    k=min(wins,losses)
    return min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/2**n)
def boot(delta):
    rng=random.Random(42)
    vals=sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(5000))
    return vals[125], vals[4874]
for split in ['validation','test']:
    print('\n====',split,'====')
    tasks=sorted(data['baseline'][split]['airline']) # placeholder
    # union keys per domain
    alldom={}
    for a in arms:
        for dom,d in data[a][split].items():
            alldom.setdefault(dom,set()).update(d)
    for a in arms:
        print('\n',a)
        for dom in sorted(alldom):
            d=data[a][split].get(dom,{})
            n=len(d); p=sum(r['passed'] for r in d.values())
            if n: print(f'  {dom:<18} {p}/{n} ({p/n:.1%})')
    print('\nPairwise overall')
    for x,y in [('mmh','flat'),('mmh','baseline'),('flat','baseline')]:
        # align union of tasks present in both; prefer domain key
        X={}; Y={}
        for dom in alldom:
            for t,r in data[x][split].get(dom,{}).items(): X[t]=r['passed']
            for t,r in data[y][split].get(dom,{}).items(): Y[t]=r['passed']
        common=sorted(set(X)&set(Y))
        wins=sum(X[t] and not Y[t] for t in common); losses=sum(Y[t] and not X[t] for t in common)
        diffs=[int(X[t])-int(Y[t]) for t in common]
        lo,hi=boot(diffs)
        print(f'  {x} vs {y}: n={len(common)} {sum(X[t] for t in common)}/{len(common)} vs {sum(Y[t] for t in common)}/{len(common)} wins={wins} losses={losses} diff={sum(diffs)/len(common):+.4f} ci95=[{lo:+.4f},{hi:+.4f}] p={mcnemar(wins,losses):.4g}')
        for dom in sorted(alldom):
            com=[t for t in common if t in data[x][split].get(dom,{}) and t in data[y][split].get(dom,{})]
            if com:
                xp=sum(X[t] for t in com); yp=sum(Y[t] for t in com)
                w=sum(X[t] and not Y[t] for t in com); l=sum(Y[t] and not X[t] for t in com)
                print(f'     {dom:<18} {xp}/{len(com)} vs {yp}/{len(com)} diff={(xp-yp)/len(com):+.3f} wins={w} losses={l}')

print('\nGuidelines files nonempty:')
for a in arms:
    for p in glob.glob(os.path.join(base,a,'evaluations','final-*','guidelines.md')):
        txt=open(p,encoding='utf-8').read().strip()
        if txt:
            print(a, os.path.basename(os.path.dirname(p)), repr(txt[:300]))