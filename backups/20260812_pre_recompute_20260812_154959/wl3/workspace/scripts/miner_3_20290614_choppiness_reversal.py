import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:return None
 d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close.rename(s)
arr=[get(s) for s in U]; p=pd.concat([x for x in arr if x is not None],axis=1).sort_index(); lr=np.log(p).diff()
# Choppiness/reversal factor: negative signed path efficiency over 20 sessions.
# It favors assets whose recent movement is less persistent, conditional on direction.
eff=lr.rolling(20,min_periods=18).sum()/lr.abs().rolling(20,min_periods=18).sum(); f=-eff
print('instruments',p.shape[1],'price_dates',len(p),'range',p.index.min(),p.index.max())
for h in [5,10,20]:
 vals=[]
 for i in range(20,len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('HORIZON',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
 for lab,m in [('2026-27',(a.index>='2026')&(a.index<'2028')),('2028+',a.index>='2028'),('recent250',np.arange(len(a))>=len(a)-250)]:
  q=a.loc[m,'ic']; print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20290614_choppiness_reversal_signal.csv',index=False)
