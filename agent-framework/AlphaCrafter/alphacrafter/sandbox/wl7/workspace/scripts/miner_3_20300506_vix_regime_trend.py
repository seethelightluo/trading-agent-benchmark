import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); lr=np.log(P).diff(); ret=P/P.shift(20)-1; vol=lr.rolling(20).std()
# In calm macro regimes, favor persistent trend; in stressed regimes, invert
# recent trend to capture cross-asset flight-to-quality/mean reversion response.
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); vs=v.set_index('date')['close'].astype(float).reindex(P.index).ffill()
q=vs.rolling(252,min_periods=60).rank(pct=True)
g=np.where(q<.35,1,np.where(q>.75,-1,0))
f=(ret/(vol+1e-12)).mul(g,axis=0).shift(1)
def calc(h):
 rows=[]
 for i in range(len(P)-h):
  fut=P.iloc[i+h]/P.iloc[i]-1; x=f.iloc[i]; ok=x.notna()&fut.notna()
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(x[ok],fut[ok]).statistic,ok.sum()))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
z=calc(10)
print('universe',len(syms),'available',len(P.columns),'dates',len(z),'avg_names',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [1,5,10,20,40]:
 q2=calc(h); print('decay',h,'IC',q2.ic.mean(),'ICIR',q2.ic.mean()/q2.ic.std(ddof=1),'dates',len(q2))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-30')]:
 q2=z.loc[a:b]; print('regime',a,b,'dates',len(q2),'IC',q2.ic.mean(),'ICIR',q2.ic.mean()/q2.ic.std(ddof=1) if len(q2)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
z.to_csv('scripts/miner_3_20300506_vix_regime_trend_ic.csv'); f.to_csv('scripts/miner_3_20300506_vix_regime_trend_signal.csv')
