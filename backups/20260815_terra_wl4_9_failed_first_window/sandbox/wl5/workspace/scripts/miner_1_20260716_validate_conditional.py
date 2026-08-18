import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().ffill().loc[:'2026-07-15']; r=p.pct_change(); v=r.rolling(20,min_periods=10).std()
f=(-p.pct_change(3)).where(v.le(v.quantile(.75,axis=1),axis=0))
# pooled rank correlations with library proxies
lib={'reversal5':-p.pct_change(5),'leadlag5':pd.DataFrame({s:p.pct_change(5).drop(columns=s).median(axis=1) for s in U}),'mom20':p.pct_change(20)/p.pct_change(20).rolling(20).std()}
for n,z in lib.items():
 q=pd.concat([f.stack().rename('f'),z.stack().rename('z')],axis=1).dropna(); print(n,round(q.f.corr(q.z),6))
# rank turnover on valid overlap
q=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',q.diff().abs().mean().mean())
# regime stats valid with spearman
from scipy.stats import spearmanr
for name,sl in [('2020-22',('2020','2022-12-31')),('2023-24',('2023','2024-12-31')),('2025-26',('2025','2026-07-15'))]:
 vals=[]
 for d in f.loc[sl[0]:sl[1]].index:
  z=pd.concat([f.loc[d].rename('f'),r.shift(-1).loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic)
 a=np.array(vals);print(name,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))
