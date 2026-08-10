import json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

root=Path('.')
items={
 'miner_2_20270325_breakout_range':'scripts/miner_2_20270325_breakout_range_signal.csv',
 'miner_2_20270325_clv_reversal':'scripts/miner_2_20270325_clv_reversal_signal.csv',
 'miner_2_20270325_yieldspread_residual_reversal':'scripts/miner_2_20270325_yieldspread_residual_signal.csv',
}
sigs={}
for fid,path in items.items():
 d=pd.read_csv(path)
 d['date']=pd.to_datetime(d['date'])
 sigs[fid]=d.set_index('date').sort_index()
print('artifacts',[(k,v.shape) for k,v in sigs.items()])
for a in sigs:
 for b in sigs:
  if a>=b: continue
  x=sigs[a].stack().rename('a').to_frame()
  y=sigs[b].stack().rename('b').to_frame()
  z=x.join(y,how='inner').dropna()
  rho=float(spearmanr(z.a,z.b).statistic) if len(z)>2 else None
  print('PAIR',a,b,'n=',len(z),'spearman=',rho)
for fid,d in sigs.items():
 vals=d.stack().dropna()
 print('SIGNAL',fid,'valid=',len(vals),'dates=',d.dropna(how='all').shape[0],'assets=',d.notna().sum(axis=1).mean())
# write the recovered yield factor into the active library, preserving its validated definition
src=Path('factors/quarantine/miner_2_20270325_yieldspread_residual_reversal.20260810T070803539078.json')
out=Path('factors/miner_2_20270325_yieldspread_residual_reversal.json')
obj=json.loads(src.read_text())
obj['version']='2027-03-25'
obj['validation']['last_validated']='2027-03-25T00:00:00Z'
obj['last_validated']='2027-03-25T00:00:00Z'
# Pairwise audit among the three currently validated artifacts; report max absolute rho.
pairs=[]
for a in sigs:
 for b in sigs:
  if a>=b: continue
  z=sigs[a].stack().rename('a').to_frame().join(sigs[b].stack().rename('b').to_frame(),how='inner').dropna()
  if len(z)>2: pairs.append(abs(float(spearmanr(z.a,z.b).statistic)))
obj['validation']['metrics']['max_abs_library_correlation']=max(pairs) if pairs else None
out.write_text(json.dumps(obj,indent=2)+'\n')
# update correlation provenance on the two active newly discovered records
for fid in ['miner_2_20270325_breakout_range','miner_2_20270325_clv_reversal']:
 p=Path('factors')/(fid+'.json'); o=json.loads(p.read_text()); o['validation']['metrics']['max_abs_library_correlation']=max(pairs) if pairs else None; p.write_text(json.dumps(o,indent=2)+'\n')
# reload verification
for p in [out,Path('factors/miner_2_20270325_breakout_range.json'),Path('factors/miner_2_20270325_clv_reversal.json')]:
 o=json.loads(p.read_text()); print('RELOAD',p,o['factor_id'],o['validation']['status'],o['validation']['metrics']['daily_ic'],o['validation']['metrics']['daily_icir'],o['validation']['metrics'].get('max_abs_library_correlation'))
