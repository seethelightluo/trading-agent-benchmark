import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END].copy()
 # Overnight move from prior close to current open; fade it for next-session return.
 overnight=d.open/d.close.shift(1)-1
 d['factor']=-overnight.shift(1)
 for h in [1,5,10]:d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows);print('symbols',x.symbol.nunique(),'period',x.date.min().date(),x.date.max().date())
for h in [1,5,10]:
 a=[];ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8:a.append(spearmanr(g.factor,g[f'y{h}']).statistic);ns.append(len(g))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']);print('coverage',round(len(v)/sum(len(z) for z in rows),4));r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 a=[]
 for dt,g in x.groupby('date'):
  if lo<=dt.year<=hi:
   g=g.dropna(subset=['factor','y1'])
   if len(g)>=8:a.append(spearmanr(g.factor,g.y1).statistic)
 a=np.array(a);print('REG',lo,hi,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
x.to_csv('scripts/miner_2_20261217_overnight_reversal_signal.csv',index=False)
