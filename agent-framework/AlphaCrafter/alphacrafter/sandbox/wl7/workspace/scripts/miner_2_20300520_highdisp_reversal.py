import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); lr=np.log(P).diff(); r5=lr.rolling(5).sum(); vol=lr.rolling(20).std()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); vv=v.set_index('date').close.astype(float).reindex(P.index).ffill()
# In stressed/high-dispersion days, short-term losers tend to rebound; otherwise remain inactive.
disp=lr.rolling(20).std().mean(axis=1); cutoff=disp.rolling(252,min_periods=60).quantile(.8)
active=(disp>cutoff).astype(float)
f=(-r5/(vol+1e-12)).mul(active,axis=0).shift(1)
def calc(h):
 out=[]
 for i in range(len(P)-h):
  y=P.iloc[i+h]/P.iloc[i]-1; x=f.iloc[i]; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append((P.index[i],spearmanr(x[ok],y[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
z=calc(10)
print('idea=highdisp_short_reversal5 universe',len(syms),'dates',len(z),'avg_names',z.n.mean(),'coverage',z.n.mean()/15,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [1,5,10,20,40]:
 q=calc(h); print('decay',h,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-20')]:
 q=z.loc[a:b]; print('regime',a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan,'hit',(q.ic>0).mean() if len(q) else np.nan)
print('rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
