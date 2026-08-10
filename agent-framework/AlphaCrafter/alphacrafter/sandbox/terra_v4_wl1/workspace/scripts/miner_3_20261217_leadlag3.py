import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allr=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 allr.append(pd.DataFrame({'date':d.date,'symbol':s,'ret':d.close.pct_change(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(allr,ignore_index=True); p=x.pivot(index='date',columns='symbol',values='ret')
# Lead-lag: yesterday's cross-sectional median return excluding the asset, volatility-normalized
med=p.median(axis=1); x['factor']=x['date'].map(med)-x['symbol'].map(p.iloc[-1].to_dict()) if False else 0.0
x['factor']=[med.get(dt,np.nan)-p.loc[dt].drop(labels=s,errors='ignore').median() for dt,s in zip(x.date,x.symbol)]
# use 3-day average of lagged market breadth/median, excluding current asset contemporaneously
p3=p.rolling(3,min_periods=2).mean().shift(1); med3=p3.median(axis=1)
x['factor']=[med3.get(dt,np.nan)-p3.loc[dt].drop(labels=s,errors='ignore').median() for dt,s in zip(x.date,x.symbol)]
def calc(df):
 vals=[]; ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8: vals.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.asarray(vals); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
print('UNIVERSE',len(syms),'rows',len(x));print('H1',calc(x))
for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026-12-17','25-26')]:print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 z=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];z.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 print('H',h,calc(x[['date','symbol','factor']].drop_duplicates().merge(pd.concat(z),on=['date','symbol'])))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v));v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_leadlag3_signal.csv',index=False)
