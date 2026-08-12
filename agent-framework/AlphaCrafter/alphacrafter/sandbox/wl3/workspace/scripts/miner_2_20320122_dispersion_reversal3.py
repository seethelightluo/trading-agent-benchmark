import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
S={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); S[s]=np.log(d.drop_duplicates('date').set_index('date').close.astype(float)).sort_index().diff()
R=pd.DataFrame(S).sort_index(); ret=R.rolling(3).sum(); vol=R.rolling(20).std(); disp=R.rolling(20).std().mean(axis=1)
# reversal is stronger when cross-sectional dispersion is high, all lagged
f=-ret/vol.replace(0,np.nan)
f=f.mul((disp/disp.rolling(120).median()).clip(.5,2),axis=0)
out=[]
for i,t in enumerate(R.index[:-10]):
 z=pd.concat([f.loc[t],R.iloc[i+1:i+11].sum()],axis=1).dropna()
 if len(z)>=8: out.append((t,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
D=pd.DataFrame(out,columns=['date','ic','n']); mu=D.ic.mean(); ir=mu/D.ic.std(ddof=1)*np.sqrt(252)
turn=[]; prev=None
for t in D.date:
 q=f.loc[t].rank(pct=True)
 if prev is not None: turn.append((q-prev).abs().mean())
 prev=q
print('dates',len(D),'avg_n',D.n.mean(),'coverage',D.n.mean()/15,'IC',mu,'ICIR',ir,'hit',(D.ic>0).mean(),'turn',np.mean(turn))
for a,b in [('2026-01-01','2029-12-31'),('2030-01-01','2032-01-22')]:
 q=D[(D.date>=a)&(D.date<=b)]; print(a,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252))
pd.DataFrame({'date':D.date.astype(str),'ic':D.ic,'n':D.n}).to_csv('scripts/miner_2_20320122_dispersion_reversal3_signal.csv',index=False)
print('artifact scripts/miner_2_20320122_dispersion_reversal3_signal.csv')
