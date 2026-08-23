import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-06')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v[v.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill()
vret=v.pct_change(3); vz=(vret-vret.rolling(60,min_periods=40).mean())/vret.rolling(60,min_periods=40).std(); stress=(vz.shift(1)>0.5)
base=-(px.pct_change(3).shift(1)); sig=base.mul(stress,axis=0).mask(~stress,axis=0); fwd=px.shift(-1)/px-1

def calc(S,F,mask=None):
 vals=[]; ns=[]
 for i,d in enumerate(S.index):
  if mask is not None and not bool(mask[i]): continue
  g=pd.DataFrame({'s':S.iloc[i].values,'f':F.iloc[i].values},index=S.columns).dropna()
  if len(g)>=8 and g.s.nunique()>1: vals.append(spearmanr(g.s,g.f).statistic); ns.append(len(g))
 a=np.array(vals); return len(a),round(np.mean(ns),2) if len(ns) else 0,round(a.mean(),6) if len(a) else None,round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None,round(np.mean(a>0),4) if len(a) else None
print('end',px.index.max().date(),'all',calc(sig,fwd)); y=sig.index.year
for q,m in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]: print(q,calc(sig,fwd,m))
print('stress dates',int(stress.sum()),'coverage',round(sig.notna().mean().mean(),4))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20271007_vix_stress_reversal_signal.csv',index=False)
