"""Durable read-only telemetry and progress for both full benchmark studies."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from .full_study import write_json, append_json, ARMS


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def snapshot(root):
    result = {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'disk_free_gb':round(shutil.disk_usage(root).free / 2**30, 1), 'models':{}}
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free',
                          '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    result['gpu'] = gpu.stdout.strip()
    for model, port in [('qwen',8000),('ling',8001)]:
        folder = root / model
        item = {'status':read(folder / 'status.json'), 'arms':{}}
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=5) as response:
                lines = response.read().decode().splitlines()
            item['server'] = [l for l in lines if l.startswith(('vllm:num_requests_running{',
                              'vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
            item['healthy'] = True
        except Exception as exc:
            item.update(healthy=False,error=str(exc))
        for arm in ARMS:
            rounds = [read(p,{}) for p in (folder / arm).glob('round-*.json')]
            tasks = [r for row in rounds for r in row.get('records',[])]
            counts = {'rounds':len(rounds),'search_scored':len(tasks),
                      'search_passed':sum(r['passed'] for r in tasks)}
            for split in ('validation','test'):
                rows = [r for p in (folder / arm / 'evaluations').glob(f'final-{split}-*/result.json')
                        for r in read(p,[])]
                counts[split+'_scored'] = len(rows)
                counts[split+'_passed'] = sum(r['passed'] for r in rows)
            item['arms'][arm] = counts
        capture = root / 'operations' / (model+'-api.jsonl')
        item['capture_bytes'] = capture.stat().st_size if capture.exists() else 0
        faults = folder / 'infrastructure.jsonl'
        item['infrastructure_attempts'] = len(faults.read_text().splitlines()) if faults.exists() else 0
        result['models'][model] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    while True:
        try:
            row = snapshot(args.root)
            write_json(args.root / 'monitor.json', row)
            append_json(args.root / 'telemetry.jsonl', row)
            print(json.dumps(row),flush=True)
        except Exception as exc:
            print(f'monitor error: {exc}',flush=True)
        if args.once:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
