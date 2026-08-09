import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[['date','close']].copy(); d['r']=d.close.pct_change()
 # inverse of failed directional efficiency: reward choppy/overextended paths likely to mean revert
 d['f']=-(d.close/d.close.shift(5)-1)/(d.r.abs().rolling(5).sum()+1e-9)
 d['symbol']=s; rows.append(d)
x=pd.concat(rows,ignore_index=True)
def validate(h):
 vals=[]
 for s,g in x.groupby('symbol'):
  g=g.sort_values('date').copy(); g['y']=g.close.shift(-h)/g.close-1; vals.append(g)
 q=pd.concat(vals); cs=[]; ns=[]; ds=[]
 for dt,g in q.groupby('date'):
  g=g.dropna(subset=['f','y'])
  if len(g)>=8:
   cs.append(spearmanr(g.f,g.y).statistic); ns.append(len(g)); ds.append(dt)
 a=np.asarray(cs,float); return len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),ds,a
for h in [1,5,10]:
 n,an,ic,ir,hit,ds,a=validate(h); print(f'h={h} dates={n} avg_names={an:.2f} IC={ic:.6f} ICIR={ir:.6f} hit={hit:.4f}')
 for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-12-31')]:
  z=a[(np.asarray(ds)>=pd.Timestamp(lo))&(np.asarray(ds)<=pd.Timestamp(hi))]
  if len(z)>1: print(' ',lo[:4]+'-'+hi[:4],len(z),f'{np.mean(z):.6f}',f'{np.mean(z)/np.std(z,ddof=1):.6f}')
wide=x.pivot(index='date',columns='symbol',values='f'); ranks=wide.rank(axis=1,pct=True)
print('coverage',f'{x.f.notna().mean():.6f}','turnover',f'{ranks.diff().abs().mean().mean():.6f}')
out=x[['date','symbol','f']].dropna(); out.to_csv('scripts/miner_1_20270114_inverse_efficiency_signal.csv',index=False)
# compare to available signal artifacts, aligned exact symbol/date, report max abs rho
corr=[]
a=out.set_index(['date','symbol']).f.rename('candidate')
import glob,os
for p in glob.glob('scripts/*_signal.csv'):
 try:
  z=pd.read_csv(p,parse_dates=['date']); cols=[c for c in z.columns if c not in ('date','symbol')]
  if len(cols)!=1 or not {'date','symbol'}<=set(z.columns): continue
  b=z.set_index(['date','symbol'])[cols[0]].rename('old'); m=pd.concat([a,b],axis=1).dropna()
  if len(m)>20:
   r=m.candidate.corr(m.old,method='spearman'); corr.append((abs(r),r,os.path.basename(p),len(m)))
 except Exception: pass
print('max_corr',max(corr) if corr else None)
