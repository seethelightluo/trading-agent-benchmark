import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv');p.date=pd.to_datetime(p.date);D[s]=p.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill().loc[:'2033-10-02'];r=px.pct_change();lag=px.shift(1);disp=r.shift(1).std(axis=1); med=disp.rolling(60).median();
# Conditional short reversal activated only in high-dispersion regimes; lagged regime.
f=(-lag.pct_change(3)).where(disp>med,0.0)
print('dates',len(px),'instruments',len(U),'coverage',round(f.notna().mean().mean(),4),'active',round((disp>med).mean(),4))
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna();print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6));f.to_csv('scripts/miner_2_20331003_highdisp_reversal_signal.csv')
