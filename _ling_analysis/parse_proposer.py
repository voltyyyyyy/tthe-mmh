import json
path = r'_ling_analysis/ling-proposer-lines.jsonl'
rows = [json.loads(l) for l in open(path, encoding='utf-8')]
print('rows', len(rows))
for i, r in enumerate(rows):
    req = r.get('request', {})
    resp = r.get('response')
    msgs = req.get('messages', [])
    user = next((m.get('content','') for m in msgs if m.get('role')=='user'), '')
    first = user.splitlines()[0] if user else ''
    content = ''
    reasoning = ''
    finish = None
    usage = None
    if isinstance(resp, dict):
        ch = (resp.get('choices') or [{}])[0]
        msg = ch.get('message') or {}
        content = msg.get('content') or ''
        reasoning = msg.get('reasoning') or msg.get('reasoning_text') or ''
        finish = ch.get('finish_reason')
        usage = resp.get('usage')
    print(f'--- {i:02d} status={r.get("status")} model={r.get("model")} first={first!r} finish={finish} usage={usage} content_len={len(content)} reasoning_len={len(reasoning)} error={r.get("error")!r}')
    print('CONTENT:', content[:800].replace('\n', ' '))
    print()