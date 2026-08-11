import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2026-07-15'
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close.rename(s)
P=pd.concat([load(s) for s in U],axis=1).sort_index().loc[:cut]
R=P.pct_change(fill_method=None); x=R.rolling(3,min_periods=3).sum(); F=-(x.sub(x.median(axis=1),axis=0)); Y=R.shift(-1)
rows=[]; sig=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('factor'),Y.loc[dt].rename('fwd')],axis=1).dropna()
 if len(z)>=8 and z.factor.nunique()>1:
  rows.append((dt,spearmanr(z.factor,z.fwd).statistic,len(z)))
  sig.append(F.loc[dt])
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.dropna()
print('dates',len(ic),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(ic)*15),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
S=pd.DataFrame(sig,index=q.index); ranks=S.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean())
for h in [5,10]:
 yy=sum(R.shift(-k) for k in range(1,h+1)); a=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:a.append(spearmanr(z.f,z.y).statistic)
 a=pd.Series(a).dropna(); print('h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std())
for name,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-07-15')]:
 a=ic.loc[lo:hi]; print(name,len(a),a.mean(),a.mean()/a.std() if len(a)>1 else np.nan)
S.to_csv('scripts/miner_1_20260730_relative_dispersion_signal.csv')
