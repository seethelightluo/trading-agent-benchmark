import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,5000)
    if x is None or len(x)<100: x=get_index_daily_data(s,5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.drop_duplicates('date').set_index('date').sort_index()
        D[s]=x['close'].astype(float)
p=pd.concat(D,axis=1).sort_index(); r=np.log(p).diff()
# daily equal-weight market residual, then 40d accumulation / 60d vol; forward 20d
m=r.mean(axis=1); resid=r.sub(m,axis=0)
fac=-(resid.rolling(40,min_periods=35).sum())/(resid.rolling(60,min_periods=45).std()*np.sqrt(40)+1e-12)
fwd=np.log(p.shift(-20)/p)
rows=[]
for dt in fac.index:
    a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# paper IC means and ICIR (sample sd)
print('assets',len(D),'dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*len(U)))
print('IC %.8f ICIR %.8f hit %.4f turnover_proxy %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean(),np.nan))
for h in [5,10,20,40]:
 f=np.log(p.shift(-h)/p); rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr),len(rr))
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
 q=x.loc[a:b,'ic']; print('regime',a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
# turnover rank signal changes
rank=fac.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna(); print('turnover',turn.mean())
# artifact
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/miner_2_20341222_residual_reversal40_signal.csv',index=False)
x.reset_index().to_csv('../persistent/miner_2_20341222_residual_reversal40_ic.csv',index=False)
