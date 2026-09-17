import json, glob, os, collections, math, random
base=r'_ling_analysis/small'
arms=['baseline','mmh','flat']
data={a:{} for a in arms} # arm -> round -> records
for a in arms:
    for p in glob.glob(os.path.join(base,a,'round-*.json')):
        r=json.load(open(p,encoding='utf-8'))
        data[a][r['round']]=r
# verify counts
for a in arms:
    rs=sorted(data[a]); rec=[x for r in rs for x in data[a][r]['records']]
    print(a,'rounds',len(rs),'records',len(rec),'passed',sum(x['passed'] for x in rec))
print('\nDOMAIN SEARCH (rounds 1-24)')
domains=[('airline',range(1,7)),('retail',range(7,13)),('telecom',range(13,19)),('banking_knowledge',range(19,25))]
for dom,rng in domains:
    print(dom)
    for a in arms:
        rec=[x for r in rng for x in data[a][r]['records']]
        print(f'  {a:<9} {sum(x["passed"] for x in rec)}/{len(rec)} ({sum(x["passed"] for x in rec)/len(rec):.1%})')
print('\nPER-ROUND PASS / INJECTED')
print('r dom               baseline             mmh                  flat')
for r in range(1,25):
    rd=data['baseline'][r]
    vals=[]
    for a in arms:
        rec=data[a][r]['records']; inj=data[a][r].get('injected',[])
        vals.append(f'{sum(x["passed"] for x in rec)}/{len(rec)} inj={inj}')
    print(f'{r:>2} {rd["domain"]:<17} {vals[0]:<20} {vals[1]:<20} {vals[2]}')
print('\nBOUNDARY (first round in each domain)')
for dom,rng in domains:
    r=list(rng)[0]
    print(dom,'round',r)
    for a in arms:
        rec=data[a][r]['records']; print(' ',a, f'{sum(x["passed"] for x in rec)}/{len(rec)}', 'passed tasks', [x['task'] for x in rec if x['passed']])