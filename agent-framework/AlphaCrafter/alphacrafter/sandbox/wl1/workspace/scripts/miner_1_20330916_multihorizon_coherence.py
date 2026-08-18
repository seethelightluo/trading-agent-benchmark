import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); px[a]=d.set_index(pd.to_datetime(d.date)).close
P=pd.DataFrame(px).sort_index().ffill()
# one completed-day lag: signal at t uses prices through t-1; forward return t+1..t+10
r10=P.pct_change(10); r40=P.pct_change(40)
vol10=P.pct_change().rolling(10).std(); vol40=P.pct_change().rolling(40).std()
# interpretable multi-horizon direction-consistency, risk scaled
sig=((r10/(vol10+1e-6)) + 0.7*(r40/(vol40+1e-6))).shift(1)
fwd=P.shift(-10)/P-1
rows=[]; turnover=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
# turnover based rank changes
R=sig.rank(axis=1,pct=True); turnover=R.diff().abs().mean(axis=1).dropna()
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*len(assets)))
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1), (z.ic>0).mean(), turnover.mean()))
for name,lo,hi in [('2020-21','2020','2021-12-31'),('2022-24','2022','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-12-31')]:
 q=z.loc[lo:hi].ic; print(name,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in sig.index:
  x=sig.loc[dt]; y=yy.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(spearmanr(x[ok],y[ok]).statistic)
 rr=pd.Series(rr); print('horizon',h,'dates',len(rr),'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1))
out=pd.DataFrame(sig); out.index.name='date'; out.to_csv('scripts/miner_1_20330916_multihorizon_coherence_signal.csv')
