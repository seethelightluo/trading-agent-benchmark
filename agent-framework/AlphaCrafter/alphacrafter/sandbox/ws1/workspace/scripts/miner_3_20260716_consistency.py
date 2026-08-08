import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2026-07-15'].ffill()
R=P.pct_change()
cands={'consistency60':R.rolling(60).mean()/R.where(R<0).rolling(60).std(),'updown60':R.clip(lower=0).rolling(60).sum()/(R.clip(upper=0).abs().rolling(60).sum()+1e-9),'range_position90':(P-P.rolling(90).min())/(P.rolling(90).max()-P.rolling(90).min())}
for name,f in cands.items():
 a=[]; tr=[]
 for i in range(100,len(P)-10):
  z=f.iloc[i]; y=R.iloc[i+1]; ok=z.notna()&y.notna()
  if ok.sum()>=8:a.append(spearmanr(z[ok],y[ok]).statistic)
  if i>100:
   q=f.iloc[i-1]; oo=z.notna()&q.notna(); tr.append((z[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
 a=np.array(a); print(name,'dates',len(a),'N',15,'IC %.5f ICIR %.5f hit %.4f turn %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean(),np.nanmean(tr)))
 for h in (5,10):
  y=P.shift(-h)/P-1; b=[]
  for i in range(100,len(P)-h):
   z=f.iloc[i]; yy=y.iloc[i];ok=z.notna()&yy.notna()
   if ok.sum()>=8:b.append(spearmanr(z[ok],yy[ok]).statistic)
  b=np.array(b); print(' ',h,'IC %.5f ICIR %.5f'%(b.mean(),b.mean()/b.std()))
