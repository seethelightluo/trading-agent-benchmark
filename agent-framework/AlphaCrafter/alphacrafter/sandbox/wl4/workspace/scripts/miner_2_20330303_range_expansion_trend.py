import os, json, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=P.pct_change()
ret20=P.pct_change(20).shift(1); v10=r.rolling(10,min_periods=8).std().shift(1); v60=r.rolling(60,min_periods=40).std().shift(1)
exp=(v10/(v60+1e-8)-1).clip(-3,3); pers=(r.gt(0).rolling(20,min_periods=15).mean().shift(1)-.5)*2
fac=(ret20/(v60*np.sqrt(20)+1e-8)*(1+.5*exp)*(.5+.5*pers)).clip(-8,8); fac=fac.sub(fac.mean(axis=1),axis=0)
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; a=[]; ns=[]
 for dt in P.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 ic=pd.Series(a).dropna(); print(json.dumps({'horizon':h,'dates':len(ic),'avg_n':round(np.mean(ns),2),'ic':round(ic.mean(),6),'icir':round(ic.mean()/ic.std(ddof=1)*np.sqrt(252),4),'hit':round((ic>0).mean(),4)}))
 for n in [260,520,780]:
  if len(ic)>=n: print('recent',n,round(ic.tail(n).mean(),6),round(ic.tail(n).mean()/ic.tail(n).std(ddof=1)*np.sqrt(252),4))
rank=fac.rank(pct=True,axis=1); print('coverage',round(fac.notna().sum().sum()/(len(P)*15),4),'turnover',round(rank.diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); rows=[]
for dt in fac.index:
 for s in U:
  if pd.notna(fac.loc[dt,s]): rows.append((dt,s,float(fac.loc[dt,s])))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/artifacts/miner_2_20330303_range_expansion_trend_signal.csv',index=False)
