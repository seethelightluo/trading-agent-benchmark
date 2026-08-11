import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2026-12-30'
def ld(s):return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut]
p=pd.concat({s:ld(s) for s in U},axis=1).sort_index().ffill();r=p.pct_change(); eq=r[U[:8]].mean(axis=1)
# trend quality penalized by rolling equity correlation; seek idiosyncratic, diversified trend
mom=p.pct_change(30); vol=r.rolling(30,min_periods=15).std(); cor=r.apply(lambda x:x.rolling(30,min_periods=15).corr(eq))
f=(mom/(vol+1e-8)*(1-cor.clip(-1,1))).shift(1)
for h in [10,20]:
 fr=p.pct_change(h).shift(-h);a=[];ns=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
 x=np.array(a);print('H',h,'dates',len(x),'avgN',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0))
 for y,g in pd.Series(x,index=ds).groupby(pd.DatetimeIndex(ds).year):print(y,g.mean(),len(g))
print('coverage',f.notna().mean().mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
