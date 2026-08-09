"""Audit reproducible signal-history coverage for each effective factor.
This is supporting infrastructure for the binding library-independence admission test."""
import glob,json,os,re
from collections import defaultdict
E=[]
for p in glob.glob('factors/*.json'):
 try:
  z=json.load(open(p))
  if z.get('validation',{}).get('status')=='EFFECTIVE': E.append((p,z.get('factor_id','')))
 except: pass
P=glob.glob('scripts/*_signal.pkl')
print('effective_records',len(E),'signal_pickles',len(P))
for p,fid in sorted(E):
 bn=os.path.basename(p).replace('.json','')
 # factor record name through first three fields, usually matches producing script date/miner prefix
 prefix='_'.join(bn.split('_')[:3])
 hits=[os.path.basename(x) for x in P if os.path.basename(x).startswith(prefix+'_')]
 exact=[x for x in hits if fid in x]
 print(('EXACT' if exact else 'DATE_MATCH' if hits else 'MISSING'),bn,'|',fid,'|',';'.join(exact or hits[:4]))
PY
python scripts/miner_1_20340928_signal_history_audit.py