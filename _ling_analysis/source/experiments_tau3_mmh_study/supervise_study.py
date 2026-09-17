"""Run four-domain preflight then the full study, preserving all failed attempts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from .full_study import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--label',choices=['qwen','ling'],required=True)
    args=parser.parse_args()
    model,port=('Qwen/Qwen3.8-27B',8102) if args.label=='qwen' else ('inclusionAI/Ling-3.0-tiny',8103)
    module=__package__+'.full_study'
    for phase in ('preflight','full'):
        out=args.root/(args.label+'-preflight' if phase=='preflight' else args.label)
        cmd=[sys.executable,'-u','-m',module,'--repo',args.repo,'--out',str(out),
             '--model',model,'--base-url',f'http://127.0.0.1:{port}/v1','--concurrency','2']
        if phase=='preflight':
            cmd.append('--smoke')
        for attempt in range(1,4):
            with (args.root/'operations'/f'{args.label}-{phase}-{attempt}.log').open('a') as log:
                result=subprocess.run(cmd,cwd=args.repo,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:
                break
            status_path=out/'status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            error=status.get('error','')
            # Only a transport/evaluation infrastructure stop is eligible for restart.
            if 'Unresolved infrastructure errors' not in error or attempt==3:
                write_json(args.root/'operations'/f'{args.label}-supervisor.json',
                           {'state':'needs_attention','phase':phase,'attempt':attempt,'error':error,
                            'returncode':result.returncode})
                return result.returncode or 1
            for _ in range(5):
                time.sleep(60)
        write_json(args.root/'operations'/f'{args.label}-supervisor.json',
                   {'state':phase+'_complete'})
    from .analyze_study import analyze
    analyze(args.root/args.label)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
