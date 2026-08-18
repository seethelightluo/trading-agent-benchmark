import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-09-17'); px={}; vv={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); d=d[d.index<=cutoff]; px[s]=d.close; vv[s]=d.volume.replace(0,np.nan)
px=pd.DataFrame(px).sort_index(); vol=pd.DataFrame(vv).reindex(px.index); r=px.pct_change(); res=r.rolling(15,min_periods=10).sum(); res=res.sub(res.median(axis=1),axis=0)
lv=np.log(vol); med=lv.rolling(40,min_periods=20).median(); mad=(lv-med).abs().rolling(40,min_periods=20).median()*1.4826; vz=((lv-med)/(mad+1e-8)).clip(-2,2)
f=(-res*(1+0.25*vz)).shift(1)
for h in [5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
fr=px.shift(-10)/px-1
for n in [365,730,1095]:
 vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('recent',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'instruments',len(U),'price_dates',len(px))
