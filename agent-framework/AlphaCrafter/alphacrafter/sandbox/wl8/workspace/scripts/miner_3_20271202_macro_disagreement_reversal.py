import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-01')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index();
mac=[]
for s in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
 x=pd.read_csv('../persistent/index_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); mac.append(x[x.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill().pct_change())
m=pd.concat(mac,axis=1); m.columns=['DXY','USDCNY','USDJPY','EURUSD','VIX']
# all standardization uses only information available before date
zs=(m-m.rolling(60,min_periods=30).mean())/m.rolling(60,min_periods=30).std(); stress=zs.abs().mean(axis=1).shift(1)
# conditional on unusually broad macro disagreement/stress; prior asset reversal is lagged
cut=stress.rolling(252,min_periods=100).quantile(.75).shift(1); mask=stress>cut
sig=-px.pct_change().shift(1); fwd=px.shift(-1)/px-1
def calc(mask):
 vals=[]; ns=[]
 for d in px.index:
  if not bool(mask.get(d,False)): continue
  g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(g))
 a=np.array(vals)
 return len(a),round(np.mean(ns),2) if len(a) else 0,round(a.mean(),6) if len(a) else np.nan,round(a.mean()/a.std(ddof=1),6) if len(a)>1 else np.nan,round((a>0).mean(),4) if len(a) else np.nan
print('end',px.index.max().date(),'dates',len(px),'highstress',calc(mask),'nonstress',calc(~mask))
y=px.index.year
for q,mm in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',px.index>=END-pd.Timedelta(days=180))]: print(q,calc(mask&mm))
out=sig.where(mask).stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20271202_macro_disagreement_reversal_signal.csv',index=False)
print('valid signal cells',int(sig.where(mask).notna().sum().sum()),'total',sig.size,'coverage',round(sig.where(mask).notna().sum().sum()/sig.size,4))
