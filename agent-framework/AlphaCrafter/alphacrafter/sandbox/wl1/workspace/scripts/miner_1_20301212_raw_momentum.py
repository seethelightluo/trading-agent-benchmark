import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2030-12-12'); D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(f);x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close
p=pd.DataFrame(D).sort_index().loc[:end].astype(float); r=p.pct_change()
for n in [5,10,20,30,40,60,90]:
 f=(p/p.shift(n)-1).shift(1); fr=p.shift(-1)/p-1; ics=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 ic=pd.Series(ics).dropna(); print(n,'dates',len(ic),'N',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'coverage',f.notna().mean().mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
