import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-22')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); ret=px.pct_change()
# Use lagged yield shock magnitude: completed-day 3d US10Y move, standardized by trailing 60d vol, then lag one day.
y=pd.read_csv('../persistent/stock_data/US10Y.csv'); y.date=pd.to_datetime(y.date); y=y[y.date<=END].set_index('date').close.sort_index()
yr=y.pct_change(); shock=(yr.rolling(3,min_periods=3).sum()/yr.rolling(60,min_periods=40).std()).shift(1).abs()
# Cross-asset 3d reversal, amplified only after unusually large yield moves.
base=-ret.rolling(3,min_periods=3).sum().shift(1); sig=base.mul(shock.reindex(px.index),axis=0)
fwd=px.shift(-1)/px-1

def calc(S,F,mask=None):
 vals=[]; ns=[]
 for d in S.index:
  g=pd.DataFrame({'s':S.loc[d],'f':F.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: vals.append(spearmanr(g.s,g.f).statistic); ns.append(len(g))
 a=np.asarray(vals)
 return {'dates':len(a),'avg_n':round(float(np.mean(ns)),2) if ns else None,'ic':round(float(a.mean()),6) if len(a) else None,'icir':round(float(a.mean()/a.std(ddof=1)),6) if len(a)>1 and a.std(ddof=1)>0 else None,'hit':round(float((a>0).mean()),4) if len(a) else None}
print('end',px.index.max().date(),'dates',len(px),'coverage',round(sig.stack().notna().mean(),4),'overall',calc(sig,fwd))
yridx=sig.index.year
for name,c in [('2020-22',yridx<=2022),('2023-25',(yridx>=2023)&(yridx<=2025)),('2026',yridx==2026),('2027',yridx==2027),('last90',sig.index>=END-pd.Timedelta(days=90)),('last180',sig.index>=END-pd.Timedelta(days=180))]: print(name,calc(sig[c],fwd[c]))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20270923_yieldshock_reversal_signal.csv',index=False)
