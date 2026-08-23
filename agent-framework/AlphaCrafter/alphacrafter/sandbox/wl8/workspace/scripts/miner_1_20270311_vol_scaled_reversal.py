import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2027-03-10'
px={}
for s in symbols:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 d=d.loc[:cut]; px[s]=d.close
ret=pd.DataFrame(px).pct_change()
# volatility-normalized one-day reversal, entirely known at t
vol=ret.rolling(20,min_periods=15).std()
f=-ret/vol
# next-day cross-sectional return
ics=[]; rows=[]; turnover=[]
prev=None
for dt in f.index:
 x=f.loc[dt].replace([np.inf,-np.inf],np.nan); y=ret.shift(-1).loc[dt]
 z=pd.concat([x.rename('f'),y.rename('y')],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.f,z.y).statistic); rows.append((dt,len(z)))
 r=x.rank(pct=True)
 if prev is not None:
  turnover.append((r-prev).abs().mean())
 prev=r
ic=np.array(ics); print('dates',len(ic),'avg_names',np.mean([x[1] for x in rows]),'coverage',np.mean([x[1] for x in rows])/15)
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'turnover',np.nanmean(turnover))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 q=np.array([v for (d,_),v in zip(rows,ics) if a<=str(d.year)<=b]); print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10]:
 vals=[]
 for dt in f.index:
  x=f.loc[dt]; y=ret.shift(-h).loc[dt]; z=pd.concat([x.rename('f'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic)
 q=np.array(vals); print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
# correlations with known factor formulas approximate
for name,g in {'rev5':-ret.rolling(5).sum(),'mom20':ret.rolling(20).sum(),'clv':((pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].close for s in symbols})-pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].low for s in symbols}))/(pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].high for s in symbols})-pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut].low for s in symbols}))).rolling(3).mean()}.items():
 a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('a'),g.loc[dt].rename('b')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.a,z.b).statistic)
 print('corr',name,np.nanmean(a))
