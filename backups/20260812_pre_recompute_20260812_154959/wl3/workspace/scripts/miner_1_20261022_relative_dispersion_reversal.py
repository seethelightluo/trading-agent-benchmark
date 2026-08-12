import pandas as pd, numpy as np, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 px[s]=d.close
px=pd.DataFrame(px).sort_index(); ret=px.pct_change(); r5=px.pct_change(5)
csmed=r5.median(axis=1); disp=r5.sub(csmed,axis=0); vol=ret.rolling(20,min_periods=12).std()
fac=(-disp/vol).clip(-4,4); fwd=px.shift(-1).div(px)-1
D=[]; rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); D.append((dt,ic,len(z)))
  rows += [(dt,s,fac.loc[dt,s]) for s in z.index]
df=pd.DataFrame(D,columns=['date','ic','n']); mean=df.ic.mean(); ir=mean/df.ic.std(ddof=1)
rank=fac.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).mean(); coverage=fac.notna().sum().sum()/(fac.shape[0]*len(U))
print({'dates':len(df),'start':str(df.date.min().date()),'end':str(df.date.max().date()),'avg_n':df.n.mean(),'IC':mean,'ICIR':ir,'hit':(df.ic>0).mean(),'coverage':coverage,'turnover':turnover})
for label,mask in [('2020-22',(df.date.dt.year<=2022)),('2023-24',df.date.dt.year.isin([2023,2024])),('2025-26',(df.date.dt.year>=2025))]:
 q=df.loc[mask,'ic']; print(label,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=pd.DataFrame(rows,columns=['date','symbol','signal']); out.to_csv('scripts/miner_1_20261022_relative_dispersion_reversal_signal.csv',index=False)
