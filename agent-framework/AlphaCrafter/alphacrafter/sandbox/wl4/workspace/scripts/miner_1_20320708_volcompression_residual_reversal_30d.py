import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
c=pd.DataFrame({s:x.close for s,x in D.items()}); r=c.pct_change()
raw=r.rolling(30,min_periods=20).sum(); resid=raw.sub(raw.mean(axis=1),axis=0)
vol20=r.rolling(20,min_periods=15).std(); vol60=r.rolling(60,min_periods=40).std()
# Mean-reversion score, strengthened during asset-level volatility compression.
compression=(1-vol20/(vol60+1e-12)).clip(-2,2)
f=(-resid/(vol60*np.sqrt(252)+1e-12)* (1+0.5*compression)).shift(1)
fr=c.shift(-10)/c-1
rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((dt,q,len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('universe',len(D),'assets',c.shape[1],'dates',len(c),'qualifying',len(z),'avg_n',round(z.n.mean(),2))
print('H10 IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
for name,sub in [('2020-2027',z.loc[:'2027']),('2028-2030',z.loc['2028':'2030']),('2031-2032',z.loc['2031':'2032']),('recent365',z.tail(365))]:
 print(name,'dates',len(sub),'IC',round(sub.ic.mean(),5),'ICIR',round(sub.ic.mean()/sub.ic.std(ddof=1),5) if len(sub)>1 else np.nan)
for h in [5,20]:
 rr=c.shift(-h)/c-1; q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 q=pd.Series(q).dropna();print('decay',h,'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
rank=f.rank(axis=1,pct=True)
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
