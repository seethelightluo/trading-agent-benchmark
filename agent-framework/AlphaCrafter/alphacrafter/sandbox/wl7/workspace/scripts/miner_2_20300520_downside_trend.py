import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); lr=np.log(P).diff()
# Downside-volatility adjusted medium trend: reward persistent gains while penalizing harmful downside variability.
r60=P/P.shift(60)-1
down=lr.where(lr<0).rolling(40,min_periods=20).std()
f=(r60/(down*np.sqrt(252)+1e-12)).shift(1)
def calc(h):
 out=[]
 for i in range(len(P)-h):
  y=P.iloc[i+h]/P.iloc[i]-1; x=f.iloc[i]; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append((P.index[i],spearmanr(x[ok],y[ok]).statistic,ok.sum()))
 z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); return z
z=calc(10)
print('idea=downside_sharpe_trend60 universe',len(syms),'dates',len(z),'avg_names',z.n.mean(),'coverage',z.n.mean()/15,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [1,5,10,20,40]:
 q=calc(h); print('decay',h,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-20')]:
 q=z.loc[a:b]; print('regime',a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
# turnover of daily cross-sectional ranks, evaluated on valid consecutive dates
r=f.rank(axis=1,pct=True); print('rank_turnover',r.diff().abs().mean(axis=1).mean())
