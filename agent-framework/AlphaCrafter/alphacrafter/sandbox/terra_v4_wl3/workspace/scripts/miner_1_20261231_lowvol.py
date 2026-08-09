import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
A=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d['r']=d.close.pct_change(); d['f']=-d.r.rolling(20).std(); d['y']=d.close.shift(-1)/d.close-1; A.append(d[['date','f','y']].assign(symbol=s))
x=pd.concat(A); cs=[]; ds=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna();
 if len(g)>=8: cs.append(spearmanr(g.f,g.y).statistic);ds.append(dt);ns.append(len(g))
z=np.array(cs);print('dates',len(z),'avg_names',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'hit',np.mean(z>0),'coverage',x.f.notna().mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=z[(np.array(ds)>=pd.Timestamp(a+'-01-01'))&(np.array(ds)<=pd.Timestamp(b+'-12-31'))];print(a,b,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
wide=x.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True);print('turnover',wide.diff().abs().mean().mean())
x[['date','symbol','f']].dropna().to_csv('scripts/miner_1_20261231_lowvol_signal.csv',index=False)
