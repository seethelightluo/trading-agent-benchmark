import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-09-18'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d['close'].loc[:cut]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(v.index)); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); V=v.reindex(dates)
vz=(V/V.rolling(60,min_periods=40).median()-1).clip(lower=0); f=(-P.pct_change(5)).mul(vz,axis=0)
def calc(start,H):
 vals=[]; ns=[]
 for i in range(start,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(ok.sum())
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): vals.append(z)
 a=np.array(vals); return len(a),a.mean(),a.mean()/a.std(ddof=1), (a>0).mean(),np.mean(ns)
for H in [1,5,10,20]:
 n,ic,ir,hit,av=calc(0,H); print(H,n,round(ic,6),round(ir,6),round(hit,4),'avgN',round(av,2))
n,ic,ir,hit,av=calc(max(0,len(P)-365-11),10); print('recent10',n,round(ic,6),round(ir,6),round(hit,4))
print('coverage',round(f.notna().mean().mean(),4),'dates',len(P),'assets',len(U),'active_rate',round((vz>0).mean(),4))
