import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['r']=d.close.pct_change(); d['f']=-d.r.shift(1); d['y']=d.close.shift(-1)/d.close-1; rows.append(d[['date','f','y']].assign(symbol=s))
x=pd.concat(rows); o=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8:o.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
a=np.array(o);print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',x.f.notna().mean())
print(x.assign(year=x.date.dt.year).groupby('year').apply(lambda z: spearmanr(z.dropna().f,z.dropna().y).statistic if len(z.dropna())>=8 else np.nan).to_string())
