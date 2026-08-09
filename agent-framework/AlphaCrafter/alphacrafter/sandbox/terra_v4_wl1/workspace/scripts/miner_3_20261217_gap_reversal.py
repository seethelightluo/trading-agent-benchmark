import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
raw={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:x['close'] for s,x in raw.items()}).ffill(); O=pd.DataFrame({s:x['open'] for s,x in raw.items()}).reindex(P.index).ffill(); H=pd.DataFrame({s:x['high'] for s,x in raw.items()}).reindex(P.index).ffill(); L=pd.DataFrame({s:x['low'] for s,x in raw.items()}).reindex(P.index).ffill(); P=P[P.index<=cut]; O=O.reindex(P.index);H=H.reindex(P.index);L=L.reindex(P.index)
# Lagged intraday gap reversal: fade yesterday's close-to-open gap, scaled by recent true-range so signals compare risk-adjusted shocks.
prev=P.shift(1); gap=(O/prev-1); tr=(H-L)/prev; scale=tr.rolling(20,min_periods=10).median().replace(0,np.nan); F=-(gap/scale).shift(0); F=F.sub(F.median(axis=1),axis=0)
F.to_csv('scripts/miner_3_20261217_gap_reversal_signal.csv',index_label='date')
R=P.pct_change()
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=pd.Series(vals); print('H',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
cs=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U);c=F.stack().corr(x.stack())
  if pd.notna(c):cs.append((abs(c),p.name,c))
 except Exception: pass
print('max_abs_library_correlation',max(cs)[0] if cs else None,sorted(cs,reverse=True)[:5]); print('period',F.index.min().date(),F.index.max().date())
