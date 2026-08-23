import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-08')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); r5=r.rolling(5,min_periods=5).sum(); r20=r.rolling(20,min_periods=20).sum()
breadth=(r20>0).mean(axis=1); reg=(breadth-0.5).apply(np.sign).replace(0,np.nan)
sig=r5.mul(reg,axis=0).shift(1); fwd=px.shift(-1)/px-1
def calc(S,F):
 vals=[]; ns=[]
 for d in S.index:
  g=pd.DataFrame({'s':S.loc[d],'f':F.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: vals.append(spearmanr(g.s,g.f).statistic); ns.append(len(g))
 a=np.asarray(vals)
 if len(a)<2:return {'dates':len(a),'avg_n':None,'ic':None,'icir':None,'hit':None}
 return {'dates':len(a),'avg_n':round(float(np.mean(ns)),2),'ic':round(float(a.mean()),6),'icir':round(float(a.mean()/a.std(ddof=1)),6),'hit':round(float((a>0).mean()),4)}
print('end',px.index.max().date(),'rows',len(px),'coverage',round(sig.stack().notna().mean(),4),'overall',calc(sig,fwd)); y=sig.index.year
for name,c in [('2020-22',y<=2022),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]: print(name,calc(sig[c],fwd[c]))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20270909_breadth_regime_momentum_signal.csv',index=False)
