import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,2500)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
# Candidate: residual short-term reversal, 3d return orthogonalized to cross-sectional median market move,
# multiplied by inverse recent idiosyncratic volatility. Higher score means expected next return.
r=p.pct_change()
med=r.median(axis=1)
# idiosyncratic residual over 20d rolling beta to median
cov=r.rolling(20).cov(med); var=med.rolling(20).var()
beta=cov.div(var,axis=0)
res=r.sub(beta.mul(med,axis=0))
res3=res.rolling(3).sum()
ivol=res.rolling(20).std()
f=-res3.div(ivol.replace(0,np.nan))
fr=p.shift(-1).div(p)-1
rows=[]
for d in f.index:
    z=f.loc[d]; y=fr.loc[d]
    q=pd.concat([z,y],axis=1).dropna()
    if len(q)>=8:
        rows.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [1,5,10]:
    yy=p.shift(-h).div(p)-1
    rr=[]
    for d in f.index:
      q=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
      if len(q)>=8: rr.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
    x=pd.Series(rr).dropna(); print('H',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'dates',len(x))
print('daily dates',len(a),'avg n',a.n.mean(),'coverage',a.n.sum()/(len(a)*len(U)))
print('regimes')
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 x=a.loc[lo:hi,'ic']; print(lo,hi,len(x),x.mean(),x.mean()/x.std(ddof=1))
# turnover rank
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean(); print('turn',turn)
# save artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20261217_resid3_signal.csv',index=False)
