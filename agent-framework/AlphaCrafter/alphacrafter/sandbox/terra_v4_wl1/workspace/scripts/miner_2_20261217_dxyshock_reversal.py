import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
m=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date');m=m[m.date<=END].set_index('date').close
mr=m.pct_change(); mz=(mr-mr.rolling(60,min_periods=40).mean())/(mr.rolling(60,min_periods=40).std()+1e-12)
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date').close.to_frame('c');r=d.c.pct_change(); z=mz.reindex(d.index).ffill().shift(1); f=-r.rolling(3).sum().shift(1)*(1+z.abs().clip(0,3)); q=pd.DataFrame({'date':d.index,'factor':f,'y1':d.c.shift(-1)/d.c-1,'y5':d.c.shift(-5)/d.c-1,'y10':d.c.shift(-10)/d.c-1}).reset_index(drop=True);rows.append(q.assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 a=[];ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8:a.append(spearmanr(g.factor,g[f'y{h}']).statistic);ns.append(len(g))
 a=np.array(a);print(h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',x.factor.notna().mean()); rr=x.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turn',rr.diff().abs().mean(axis=1).mean())
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_2_20261217_dxyshock_reversal_signal.csv',index=False)
