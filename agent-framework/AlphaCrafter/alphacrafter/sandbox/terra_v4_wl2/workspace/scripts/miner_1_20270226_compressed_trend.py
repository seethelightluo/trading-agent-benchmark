import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in syms:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
rows=[]
for s,d in D.items():
 c=d.close.astype(float); hi=d.high.astype(float); lo=d.low.astype(float)
 ret=c.pct_change(15); rng=((hi-lo)/c).rolling(20,min_periods=15).mean(); f=ret/(rng+1e-8)
 # lag factor one date, forward 5d
 fr=c.shift(-5)/c-1
 for dt in c.index: rows.append((dt,s,f.loc[dt],fr.loc[dt]))
x=pd.DataFrame(rows,columns=['date','sym','f','r']).replace([np.inf,-np.inf],np.nan).dropna(); rec=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1: rec.append((dt,spearmanr(g.f,g.r).statistic,g))
a=np.array([z[1] for z in rec]); print('dates',len(a),'avgN',np.mean([len(z[2]) for z in rec]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for n,lo,hi in [('pre','2020','2026-07-16'),('online','2026-07-16','2027-02-26'),('recent','2026-01-01','2027-02-26')]:
 q=np.array([z for dt,z,_ in rec if str(dt)>=lo and str(dt)<hi]);print(n,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
w=x.pivot(index='date',columns='sym',values='f'); print('coverage',len(x)/(15*len(set(x.date))),'turnover',w.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean());w.to_csv('../persistent/factor_signals_miner_1_20270226_compressed_trend.csv')
