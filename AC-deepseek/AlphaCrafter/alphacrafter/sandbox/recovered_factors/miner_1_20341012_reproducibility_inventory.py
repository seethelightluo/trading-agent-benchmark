"""Inventory whether each admitted factor has a reproducible producer script,
and whether that script can be identified by its factor-definition stem.

This does not infer signal equivalence from a loosely similar filename: it reports
only exact stem / factor-id evidence needed before re-executing producers.
"""
import glob, json, os, re

def tokens(s):
    return set(x for x in re.split(r'[^a-z0-9]+', s.lower()) if len(x) > 2

records=[]
for p in glob.glob('factors/*.json'):
    try:
        d=json.load(open(p,encoding='utf-8'))
        if d.get('validation',{}).get('status')=='EFFECTIVE':
            records.append((p,d))
    except Exception: pass
scripts=glob.glob('scripts/*.py')
pickles=glob.glob('scripts/*_signal.pkl')
print('effective_records',len(records),'producer_scripts',len(scripts),'serialized_signals',len(pickles))
repro=0
for path,d in sorted(records):
    fid=d['factor_id']; stem=os.path.basename(path)[:-5]
    # Exact factors IDs embedded in code are valid producer evidence.
    exact_code=[]
    for s in scripts:
        try:
            txt=open(s,encoding='utf-8').read()
            if fid in txt: exact_code.append(os.path.basename(s))
        except Exception: pass
    # A pkl is direct evidence only when its basename includes full factor id.
    exact_pkl=[os.path.basename(x) for x in pickles if fid in os.path.basename(x)]
    # Report best lexical candidate merely as an investigation lead, never evidence.
    st=tokens(fid); scored=[]
    for s in scripts:
        q=tokens(os.path.basename(s)); scored.append((len(st&q)/max(1,len(st|q)),os.path.basename(s)))
    best=max(scored) if scored else (0,'NONE')
    ok=bool(exact_code and exact_pkl)
    repro+=ok
    print(('REPRODUCIBLE' if ok else 'BLOCKED'),fid,
          'code='+('|'.join(exact_code[:3]) or 'NONE'),
          'pkl='+('|'.join(exact_pkl[:3]) or 'NONE'),
          'lead=%.3f:%s'%best)
print('directly_reproducible_histories',repro,'of',len(records))
