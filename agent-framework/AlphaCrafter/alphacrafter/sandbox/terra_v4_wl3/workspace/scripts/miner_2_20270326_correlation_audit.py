import json, glob, os
import pandas as pd

files = {
 'yieldspread':'scripts/miner_2_20270325_yieldspread_residual_signal.csv',
 'vix5':'scripts/miner_2_20270325_vix_conditional_5d_reversal_signal.csv',
 'vixclv':'scripts/miner_3_20270325_vix_conditioned_reversal_signal.csv',
}
long={}
for name,path in files.items():
    d=pd.read_csv(path)
    # normalize the two common layouts
    if 'date' not in d.columns:
        d=d.rename(columns={d.columns[0]:'date'})
    d['date']=pd.to_datetime(d['date'])
    d=d.set_index('date')
    d=d.apply(pd.to_numeric,errors='coerce')
    long[name]=d.stack().rename(name)
x=pd.concat(long.values(),axis=1).dropna(how='all')
print('artifacts', {k:len(v) for k,v in long.items()}, 'aligned cells',len(x))
print('pairwise pearson')
for a in files:
  for b in files:
    if a<b:
      z=x[[a,b]].dropna(); print(a,b,'n',len(z),'rho',z[a].corr(z[b]))
# Cross-sectional rank correlation by date, then average of date ICs
for a in files:
  for b in files:
    if a<b:
      z=pd.concat([long[a],long[b]],axis=1).dropna()
      vals=[]
      for dt,g in z.groupby(level=0):
        if len(g)>=8: vals.append(g.iloc[:,0].rank().corr(g.iloc[:,1].rank()))
      print('daily rank',a,b,'dates',len(vals),'mean',sum(vals)/len(vals),'maxabs',max(abs(v) for v in vals))
# update provenance metrics for effective jsons
updates={
 'factors/miner_2_20270325_vix_conditional_5d_reversal.json': max(abs(x['vix5'].corr(x[c])) for c in ['yieldspread','vixclv']),
 'factors/miner_3_20270325_vix_conditioned_reversal.json': max(abs(x['vixclv'].corr(x[c])) for c in ['yieldspread','vix5']),
}
for path,val in updates.items():
    j=json.load(open(path)); j.setdefault('validation',{}).setdefault('metrics',{})['max_abs_library_correlation']=float(val)
    j['validation']['correlation_audit']={'method':'aligned signal-artifact Pearson correlation','artifacts':list(files.values()),'audited_at':'2027-03-26T00:00:00Z'}
    with open(path,'w') as f: json.dump(j,f,indent=2)
    print('UPDATED',path,val)
# verify reload
for path in updates:
 j=json.load(open(path)); print('VERIFY',os.path.basename(path),j['factor_id'],j['validation']['status'],j['validation']['metrics']['max_abs_library_correlation'])
