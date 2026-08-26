import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-10-14')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in syms}).sort_index().loc[:cut]
r=p.pct_change(); total=r.rolling(20,min_periods=15).std(); down=r.where(r<0).rolling(20,min_periods=15).std()
# Downside-volatility-weighted 10-day reversal: emphasize assets whose recent risk was downside-led.
sig=-p.pct_change(10)*(down/total)
rows=[]; art=[]
for d in sig.index:
    f=p.shift(-10).loc[d]/p.loc[d]-1
    z=pd.concat([sig.loc[d].rename('x'),f.rename('y')],axis=1).dropna()
    if len(z)>=8:
        rows.append((d,len(z),spearmanr(z.x,z.y).statistic))
        art += [(d,s,float(sig.loc[d,s]),float(f[s])) for s in z.index]
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('candidate=downside_vol_weighted_reversal_10 cutoff=2034-10-14')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for x,y in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 q=a.loc[x:y].ic; print('regime',x,y,'dates',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [1,5,10,20]:
 v=[]
 for d in sig.index:
  f=p.shift(-h).loc[d]/p.loc[d]-1; z=pd.concat([sig.loc[d].rename('x'),f.rename('y')],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.x,z.y).statistic)
 print('horizon',h,'dates',len(v),'IC',np.mean(v),'ICIR',np.mean(v)/np.std(v,ddof=1))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('artifact rows',len(art))
pd.DataFrame(art,columns=['date','symbol','signal','forward_10d_return']).to_csv('scripts/miner_1_20341030_downside_vol_reversal_signal.csv',index=False)
