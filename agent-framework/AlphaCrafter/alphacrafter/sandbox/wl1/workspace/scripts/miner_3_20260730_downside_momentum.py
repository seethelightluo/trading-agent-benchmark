import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index()
r=p.pct_change(); w=20
# downside-risk-adjusted momentum: trailing return divided by downside deviation, rewarding trends with limited negative shocks
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(w,len(p)):
 h=r.iloc[i-w:i]
 down=h.where(h<0)
 dd=np.sqrt((down**2).mean()).replace(0,np.nan)
 f.iloc[i]=(p.iloc[i]/p.iloc[i-w]-1)/dd
for horizon in [1,5,10]:
 ic=[]; ns=[]
 for i in range(len(p)-horizon):
  q=pd.concat([f.iloc[i],(p.iloc[i+horizon]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
 x=np.array(ic); print('horizon',horizon,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
vals=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(q)>=8: vals.append((f.index[i],spearmanr(q.iloc[:,0],q.y).statistic))
z=pd.Series(dict(vals)); print('assets',len(U),'valid_dates',len(z),'regime', {int(y):round(z[z.index.year==y].mean(),5) for y in sorted(z.index.year.unique())})
# rank turnover
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i].rank(pct=True); b=f.iloc[i-1].rank(pct=True)
 if a.notna().sum()>=8 and b.notna().sum()>=8: turn.append(np.abs(a-b).mean())
print('rank_turnover',round(np.mean(turn),6))
