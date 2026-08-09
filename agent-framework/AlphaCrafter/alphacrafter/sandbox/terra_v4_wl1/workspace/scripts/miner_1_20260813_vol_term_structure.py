import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'] for s in U}).sort_index(); r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); v120=r.rolling(120,min_periods=80).std(); f=-(v20/v120)
def calc(i,h):
 a=f.iloc[i].rename('f'); b=(p.iloc[i+h]/p.iloc[i]-1).rename('y'); z=pd.concat([a,b],axis=1).dropna(); return len(z), spearmanr(z.f,z.y).statistic
rows=[]
for i in range(120,len(p)-10):
 n,ic=calc(i,1)
 if n>=8: rows.append((p.index[i],n,ic))
d=pd.DataFrame(rows,columns=['date','n','ic']); d.date=pd.to_datetime(d.date); d=d.set_index('date'); print('dates',len(d),'avgN',d.n.mean(),'coverage',d.n.mean()/15)
for h in [1,5,10]:
 q=pd.Series([calc(i,h)[1] for i in range(120,len(p)-h) if calc(i,h)[0]>=8]); print('H',h,'N',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=d[(d.index.year>=lo)&(d.index.year<=hi)].ic; print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()); print('cutoff',cutoff)
