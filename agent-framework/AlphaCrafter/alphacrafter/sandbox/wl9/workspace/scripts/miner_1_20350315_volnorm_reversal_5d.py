import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>200}).sort_index().ffill()
r=px.pct_change(); vol=r.rolling(20).std(); f=(-r.rolling(5).sum()/vol).shift(1)
for h in [10,20,40,60]:
  vals=[]
  for i in range(1,len(px)-h):
    x=f.iloc[i]; y=px.iloc[i+h]/px.iloc[i]-1; ok=x.notna()&y.notna()
    if ok.sum()>=8: vals.append(float(x[ok].corr(y[ok])))
  a=np.asarray(vals); ic=np.nanmean(a); icir=ic/np.nanstd(a,ddof=1)*np.sqrt(len(a))
  print('H',h,'dates',len(a),'avgN',round(float((f.notna().sum(axis=1)).mean()),2),'IC',round(float(ic),6),'ICIR',round(float(icir),6),'hit',round(float((a>0).mean()),4))
coverage=float((f.notna().sum(axis=1)/len(U)).mean()); ranks=f.rank(pct=True,axis=1); turnover=float(ranks.diff().abs().mean(axis=1).dropna().mean())
print('coverage',round(coverage,4),'turnover',round(turnover,4),'instruments',len(px.columns),'dates',len(px),'range',px.index.min(),px.index.max())
for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-03-14')]:
 vals=[]
 for i in range(1,len(px)-10):
  if not (px.index[i]>=pd.Timestamp(a) and px.index[i]<=pd.Timestamp(b)): continue
  x=f.iloc[i]; y=px.iloc[i+10]/px.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(float(x[ok].corr(y[ok])))
 q=np.asarray(vals); print('REG',a[:4],b[:4],'dates',len(q),'IC',round(float(np.nanmean(q)),6),'ICIR',round(float(np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q))),6))
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20350315_volnorm_reversal_5d_signal.csv')
