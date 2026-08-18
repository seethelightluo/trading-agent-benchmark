# 60-day downside/upside volatility balance candidate
import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 fn='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(fn); dc='date' if 'date' in d else d.columns[0]; cc='close' if 'close' in d else 'Close'; px[s]=pd.Series(d[cc].values,index=pd.to_datetime(d[dc]))
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); w=60
up=r.where(r>0,0.).rolling(w,min_periods=40).std(); dn=(-r.where(r<0,0.)).rolling(w,min_periods=40).std()
f=np.log((up/(dn+1e-8)).clip(.1,10)).shift(1); f=f.sub(f.mean(axis=1),axis=0); y=p.shift(-10)/p-1
ics=[]; sig=[]
for dt in f.index:
 ok=f.loc[dt].notna()&y.loc[dt].notna()
 if ok.sum()>=8:
  ics.append((dt,f.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'),ok.sum()))
  sig += [(dt,s,float(f.loc[dt,s])) for s in U if ok.get(s,False)]
z=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date').loc['2020':'2033-10-14']; m=z.ic.mean(); sd=z.ic.std(ddof=1)
print('dates',len(z),'N',z.n.mean(),'IC',m,'ICIR',m/sd,'annualized',m/sd*np.sqrt(252),'hit',(z.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=z.loc[a:b].ic
 if len(q)>1: print(a,len(q),q.mean(),q.mean()/q.std(ddof=1))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20331014_asymmetry60_signal.csv',index=False)
