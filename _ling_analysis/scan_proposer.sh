#!/bin/bash
set -eu
F=/home/yangfan/mmh-exp/full375-20260916-v2/operations/ling-api.jsonl
M='You maintain a small set of behavioural guidelines for a customer-service agent.'
echo "marker_count: $(grep -F -c "$M" "$F" || true)"
grep -F "$M" "$F" > /tmp/ling-proposer-lines.jsonl || true
wc -l /tmp/ling-proposer-lines.jsonl
ls -lh /tmp/ling-proposer-lines.jsonl