import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
files={os.path.basename(x).replace('.csv',''):x for x in glob.glob('../persistent/stock_data/*.csv')}
rows=[]
for s in syms:
 p=files.get(s)
 if not p: continue
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 d['r']=d.close.pct_change(); vol=d.volume.replace(0,np.nan)
 # Trend continuation only when prior volume exceeds its trailing median; all information is lagged one bar.
 shock=(vol.shift(1)/(vol.shift(2).rolling(20,min_periods=10).median()+1e-12)-1).clip(lower=0)
 trend=d.close.shift(1).pct_change(20)
 d['factor']=trend*np.log1p(shock)
 for h in [1,5,10]: d[f'fwd{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','fwd1','fwd5','fwd10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'fwd{h}'])
  if len(g)>=8: obs.append(spearmanr(g.factor,g[f'fwd{h}']).statistic); ns.append(len(g))
 a=np.array(obs); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),5),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/sum(len(z) for z in rows),4))
ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().mean(axis=1).mean(),4))
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
