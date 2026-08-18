import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-11-12'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close[d.index<=cutoff]
px=pd.DataFrame(px).sort_index(); r=px.pct_change(); ret20=px.pct_change(20); path=r.abs().rolling(20,min_periods=15).sum(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Reversal of directional return, penalized by path inefficiency and scaled by volatility; lagged
f=(-(ret20/(path+1e-8))/(vol+1e-8)).shift(1)
def calc(h, dates=None):
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 it=f.index if dates is None else f.index[-dates:]
 for dt in it:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 x=pd.Series(vals); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
for h in [5,10,20]: print('H',h,'dates avgN IC ICIR hit',*(round(x,6) if isinstance(x,float) else x for x in calc(h)))
for n in [365,730,1095]:
 z=calc(10,n);print('recent',n,'dates',z[0],'IC',round(z[2],6),'ICIR',round(z[3],6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U),'cutoff',cutoff.date())
