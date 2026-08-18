import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
  D[s]=x
# common daily panel
close=pd.DataFrame({s:d.close for s,d in D.items()}); high=pd.DataFrame({s:d.high for s,d in D.items()}); low=pd.DataFrame({s:d.low for s,d in D.items()})
# range-location: low position gets positive score; only completed t data
hh=high.rolling(20,min_periods=15).max(); ll=low.rolling(20,min_periods=15).min()
loc=(close-ll)/(hh-ll).replace(0,np.nan)
f=0.5-loc
# rank-normalized across section is equivalent for IC; forward return
r=close.shift(-10)/close-1
rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],r.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  ic=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
  rows.append((dt,ic,len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(x): return (len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1), (x.ic>0).mean())
print('dates',len(z),'avg_n',z.n.mean(),'min_n',z.n.min(),'coverage',z.n.sum()/(len(z)*len(U)))
print('H10 n IC ICIR hit',stat(z))
for name,sub in [('2020-2023',z.loc[:'2023']),('2024-2027',z.loc['2024':'2027']),('2028-2032',z.loc['2028':'2032'])]: print(name,stat(sub))
# decay
for h in [5,20]:
 rr=close.shift(-h)/close-1; q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 q=pd.Series(q).dropna(); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
# turnover based on cross-sectional ranks on adjacent valid dates
rank=f.rank(axis=1,pct=True); print('turnover_proxy',rank.diff().abs().mean(axis=1).mean())
