import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in S:
 f='../persistent/stock_data/'+s+'.csv'; px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index().loc[:'2027-10-06']
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# One-day reversal with close-location/range confirmation: recent shock is stronger when close is near the day's extreme.
d=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2027-10-06'] for s in S},axis=1)
clv=pd.DataFrame({s:((d[s].close-d[s].low)/(d[s].high-d[s].low).replace(0,np.nan)-.5) for s in S})
f=((-r/vol)*(1+clv.abs())).clip(-8,8).shift(1).ewm(span=3,min_periods=3,adjust=False).mean(); y=p.shift(-1)/p-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=o.ic
print('dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(len(o)*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [3,5,10]:
 yy=p.shift(-h)/p-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=pd.Series(a); print('h',h,a.mean(),a.mean()/a.std(ddof=1),len(a))
