import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; rows.append(d[['date','close']].assign(symbol=s))
x=pd.concat(rows); p=x.pivot(index='date',columns='symbol',values='close'); r=p.pct_change(); market=r.median(axis=1); asset3=p.pct_change(3); m3=market.rolling(3).sum(); f=-(asset3.sub(m3,axis=0)); out=f.stack().rename('factor').reset_index().rename(columns={'level_1':'symbol'}); px=p.stack().rename('close').reset_index().rename(columns={'level_1':'symbol'}); out=out.merge(px,on=['date','symbol']); fut=p.shift(-1).stack().rename('future').reset_index().rename(columns={'level_1':'symbol'}); out=out.merge(fut,on=['date','symbol']); out['y1']=out.future/out.close-1
for h in [5,10]:
 fut=p.shift(-h).stack().rename('fh').reset_index().rename(columns={'level_1':'symbol'}); z=out.merge(fut,on=['date','symbol']);out[f'y{h}']=z.fh/z.close-1

def calc(z,y):
 a=[];ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor',y]);
  if len(g)>=8:
   q=spearmanr(g.factor,g[y]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(g))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('universe',len(syms),'rows',len(out),'date span',out.date.min(),out.date.max());
for h in ['y1','y5','y10']:print(h,calc(out,h))
for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026','25-26')]:print(n,calc(out[(out.date>=lo)&(out.date<=hi)],'y1'))
print('coverage',out.factor.notna().mean(), 'valid rows',out.factor.notna().sum())
ranks=out.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',ranks.diff().abs().mean(axis=1).mean());out[['date','symbol','factor']].dropna().to_csv('scripts/miner_1_20261217_market_relative_reversal_signal.csv',index=False)
