import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); lr=np.log(P).diff(); r=P.pct_change()
# Volatility-shock reversal: reverse recent 5d return, amplified when short-run
# realized volatility is elevated relative to its 60d baseline; lagged one day.
v5=lr.rolling(5).std(); v60=lr.rolling(60).std()
f=(-r.rolling(5).sum()*(v5/(v60+1e-10))).shift(1)
def run(h):
 out=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append((P.index[i],spearmanr(x[ok],y[ok]).statistic,ok.sum()))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
z=run(10)
print('universe',15,'dates',len(z),'avg_names',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [1,5,10,20,40]:
 q=run(h); print('decay',h,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'dates',len(q))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-15')]:
 q=z.loc[a:b]; print('regime',a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean())
z.to_csv('scripts/miner_3_20300422_vol_shock_reversal_ic.csv'); f.to_csv('scripts/miner_3_20300422_vol_shock_reversal_signal.csv')
