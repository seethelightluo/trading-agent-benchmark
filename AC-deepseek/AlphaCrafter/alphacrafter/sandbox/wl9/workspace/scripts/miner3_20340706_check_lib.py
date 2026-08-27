"""Check status of all factors in library"""
import json, glob, os

files = sorted(glob.glob("factors/*.json"))
for f in files:
    if 'ense' in f or 'evicted' in f:
        continue
    try:
        d = json.load(open(f))
        vid = d.get('factor_id','?')
        vstat = d.get('validation',{}).get('status','?')
        vic = d.get('validation',{}).get('metrics',{}).get('IC','?')
        vicir = d.get('validation',{}).get('metrics',{}).get('ICIR','?')
        vdate = d.get('last_validated','?')
        tags = d.get('tags',[])
        print(f"{vid:30s} status={vstat:12s} IC={str(vic):8s} ICIR={str(vicir):8s} last_val={vdate} tags={tags}")
    except Exception as e:
        print(f"ERROR reading {f}: {e}")
