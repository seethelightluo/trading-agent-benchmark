import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-11-17');P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index();r=px.pct_change(); lag=r.shift(1)
# Lagged 5d trend activated by compressed 20d volatility relative to its 60d history.
trend=px.pct_change(5).shift(1); vol=r.rolling(20).std().shift(1); baseline=vol.rolling(60).median().shift(1)
compression=(vol/baseline).clip(lower=.25,upper=2.0)
sig=trend/(compression+1e-9)
fwd=px.shift(-1)/px-1
def calc(mask=None):
 a=[];ns=[]
 for i,d in enumerate(sig.index):
  if mask is not None and not bool(mask[i]):continue
  g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   z=spearmanr(g.s,g.f).statistic
   if np.isfinite(z):a.append(z);ns.append(len(g))
 a=np.array(a)
 return len(a),round(np.mean(ns),2),round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1),6),round(np.mean(a>0),4)
y=sig.index.year;print('end',px.index.max().date(),'overall',calc())
for q,m in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]:print(q,calc(m))
print('coverage',int(sig.notna().sum().sum()),'of',sig.size,'dates',sig.index.size)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271118_volatility_compression_signal.csv',index=False)
