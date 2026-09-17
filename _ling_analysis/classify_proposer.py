import json, re, math, sys
rows = [json.loads(l) for l in open(r'_ling_analysis/ling-proposer-lines.jsonl', encoding='utf-8')]

def arm_order(r):
    if r % 3 == 2:
        return ['flat','mmh']
    return ['mmh','flat']

def extract_round(user):
    m = re.search(r'Round\s+(\d+)\s+failures', user or '')
    return int(m.group(1)) if m else None

def try_classify(content):
    txt = (content or '').strip()
    if not txt:
        return ('empty', '')
    if not txt.startswith('{'):
        return ('not_raw_json', txt[:40].replace('\n',' '))
    try:
        payload = json.loads(txt)
    except Exception as e:
        return ('json_error', str(e)[:80])
    op = str(payload.get('operation',''))
    target = payload.get('target_rule_id')
    nr = payload.get('new_rule') or {}
    if op.lower() in ('none',''):
        return ('operation_none', f'target={target!r}')
    if op.upper() != 'ADD':
        return ('op_not_add', op)
    if not isinstance(target,str) or target.lower()!='new':
        return ('add_target_not_new', repr(target))
    # validate fields like parse_patch_response
    for field in ('phi','psi','omega'):
        v = nr.get(field) if isinstance(nr,dict) else None
        if not isinstance(v,str) or not v.strip():
            return ('bad_new_rule', f'{field}={v!r}')
    conf = nr.get('confidence') if isinstance(nr,dict) else None
    try:
        conf = float(conf)
    except Exception:
        return ('bad_new_rule', f'confidence={conf!r}')
    if not (0 <= conf <= 1):
        return ('bad_new_rule', f'confidence_out_of_range={conf!r}')
    lifespan = nr.get('lifespan') if isinstance(nr,dict) else None
    if isinstance(lifespan, bool) or not isinstance(lifespan,int) or lifespan < 0:
        return ('bad_new_rule', f'lifespan={lifespan!r}')
    jc = payload.get('judge_confidence', nr.get('confidence'))
    try:
        jc = float(jc)
    except Exception:
        return ('bad_judge_confidence', repr(jc))
    if not (0 <= jc <= 1):
        return ('bad_judge_confidence', repr(jc))
    return ('valid_add_new', f'phi_len={len(nr["phi"])} psi_len={len(nr["psi"])} conf={conf} judge={jc}')

print('idx round arm  class                 detail')
for i,r in enumerate(rows):
    req = r.get('request', {})
    user = next((m.get('content','') for m in req.get('messages',[]) if m.get('role')=='user'), '')
    rd = extract_round(user)
    order = arm_order(rd) if rd else ['?','?']
    pair_idx = i - ( (rd-1)*2 if rd else 0 )
    arm = order[pair_idx] if 0 <= pair_idx < len(order) else '?'
    resp = r.get('response')
    content = ''
    if isinstance(resp,dict):
        msg = (resp.get('choices') or [{}])[0].get('message') or {}
        content = msg.get('content') or ''
    cls, detail = try_classify(content)
    print(f'{i:02d}  r{rd:<2d}  {arm:<4s} {cls:<20s} {detail}')