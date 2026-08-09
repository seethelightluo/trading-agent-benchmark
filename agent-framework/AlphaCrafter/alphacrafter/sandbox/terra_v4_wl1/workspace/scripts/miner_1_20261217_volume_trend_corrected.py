import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 p='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); v=d.volume.replace(0,np.nan)
 shock=(v.shift(1)/(v.shift(2).rolling(20,min_periods=10).median())-1).clip(lower=0)
 d['factor']=d.close.shift(1).pct_change(20)*np.log1p(shock)
 d['fwd1']=d.close.shift(-1)/d.close-1; d['fwd5']=d.close.shift(-5)/d.close-1; d['fwd10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','fwd1','fwd5','fwd10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  z=g.dropna(subset=['factor',f'fwd{h}'])
  if len(z)>=8 and z.factor.nunique()>1 and z[f'fwd{h}'].nunique()>1:
   q=spearmanr(z.factor,z[f'fwd{h}']).statistic
   if np.isfinite(q): a.append(q); ns.append(len(z))
 a=np.array(a); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4))
print('coverage',round(x.factor.notna().mean(),4),'symbols',x.symbol.nunique(),'period',x.date.min(),x.date.max())
# turnover among valid ranks
p=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',round(p.diff().abs().mean(axis=1).mean(),4))
for y,g in x.groupby(x.date.dt.year):
 a=[]
 for _,z in g.groupby('date'):
  z=z.dropna(subset=['factor','fwd1'])
  if len(z)>=8 and z.factor.nunique()>1 and z.fwd1.nunique()>1:a.append(spearmanr(z.factor,z.fwd1).statistic)
 print('year',y,'dates',len(a),'ICIR',round(np.mean(a)/np.std(a,ddof=1),4) if len(a)>1 else np.nan)
