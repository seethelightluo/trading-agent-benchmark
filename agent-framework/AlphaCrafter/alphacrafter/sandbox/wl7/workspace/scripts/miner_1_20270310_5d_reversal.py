import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    x=get_stock_daily_data(s,2200)
    if x is None or len(x)<100: x=get_index_daily_data(s,2200)
    return x
D={}
for s in U:
    x=get(s)
    if x is not None:
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date')['close'].astype(float); D[s]=x
P=pd.DataFrame(D).sort_index(); r=P.pct_change()
# 5-session reversal, lagged: signal at t based through t-1
f=-(P.pct_change(5).shift(1))/r.rolling(20).std().shift(1)
fr=P.pct_change().shift(-1)
rows=[]
for d in P.index:
    a=f.loc[d]; b=fr.loc[d]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8: rows.append([d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)])
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'assets',len(D),'coverage',q.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for h in [1,5,10,20]:
 ff=P.pct_change(h).shift(-h)
 rr=[]
 for d in P.index:
  z=pd.concat([f.loc[d],ff.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr),len(rr))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,len(z),z.mean(),z.mean()/z.std() if len(z)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20270310_5d_reversal_signal.csv',index=False)
