import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in syms}).sort_index(); r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv').set_index('date')['close'].reindex(p.index).ffill(); dxy=pd.read_csv('../persistent/index_data/DXY.csv').set_index('date')['close'].reindex(p.index).ffill()
m=r.mean(axis=1); dv=vix.pct_change(); dd=dxy.pct_change()
# Acute joint stress: VIX jump is unusually large, with simultaneous dollar strengthening.
# Threshold uses only rolling history and avoids look-ahead.
zv=(dv-dv.rolling(60).mean())/dv.rolling(60).std(); zd=(dd-dd.rolling(60).mean())/dd.rolling(60).std(); event=(zv>1.0)&(zd>0.25)
# rolling conditional event return/beta, requiring 10 events within trailing 120 sessions
f=pd.DataFrame(index=p.index,columns=syms,dtype=float)
for s in syms:
 y=r[s]; ev=event.astype(float)
 cnt=ev.rolling(120).sum(); ey=(y*ev).rolling(120).sum()/cnt
 em=(m*ev).rolling(120).sum()/cnt; evar=((m-em)**2*ev).rolling(120).sum()/cnt
 cov=((y-ey)*(m-em)*ev).rolling(120).sum()/cnt
 eb=cov/evar
 # reward positive event return, penalize stress beta; residualize ordinary beta
 om=m.rolling(80).mean(); ov=m.rolling(80).var(); ob=((y-m.rolling(80).mean())*(m-om)).rolling(80).mean()/ov
 f[s]=ey-0.5*eb-0.25*ob
f=f.sub(f.mean(axis=1),axis=0)
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
rank=f.rank(axis=1,pct=True);print('source_dates',len(p),'instruments',len(syms),'events',int(event.sum()),'coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum(),'turnover',rank.diff().abs().mean().mean())
