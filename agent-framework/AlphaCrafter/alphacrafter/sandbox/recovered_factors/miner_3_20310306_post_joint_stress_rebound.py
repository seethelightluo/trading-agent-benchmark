import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in syms}).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
vix=pd.read_csv('../persistent/index_data/VIX.csv').set_index('date')['close'].reindex(p.index).ffill(); dxy=pd.read_csv('../persistent/index_data/DXY.csv').set_index('date')['close'].reindex(p.index).ffill()
dv=vix.pct_change(); dd=dxy.pct_change(); zv=(dv-dv.rolling(60).mean())/dv.rolling(60).std(); zd=(dd-dd.rolling(60).mean())/dd.rolling(60).std()
event=((zv>1.0)&(zd>0.25)).astype(float)
# Event-response reversal: after a joint stress day, assets with unusually poor subsequent 5-session response tend to rebound.
# At date t, use only completed event responses whose response window ended by t.
resp=pd.DataFrame(index=p.index,columns=syms,dtype=float)
for s in syms: resp[s]=r[s].rolling(5).sum().shift(-4)
# shift response one day so at t it excludes a response ending after t; event at t-5 has response through t-1
known=resp.shift(1)
num=(known.mul(event,axis=0)).rolling(240).sum(); den=event.rolling(240).sum(); f=num.div(den,axis=0)
# require at least 8 events; cross-sectional demean
f[den<8]=np.nan; f=f.sub(f.mean(axis=1),axis=0)
dates=p.index
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[];ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'meanIC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'meanN',np.mean(ns))
fr=p.shift(-1)/p-1; out=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: out.append((dates[i],spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
v=pd.DataFrame(out,columns=['date','ic']).set_index('date')
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2030-08','2031-02')]:
 q=v.loc[a:b,'ic'];print('REG',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
rank=f.rank(axis=1,pct=True); print('source_dates',len(p),'instruments',len(syms),'events',int(event.sum()),'coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum(),'turnover',rank.diff().abs().mean().mean())
print('event dates',event.sum())
