import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   z=fn(s,5000)
   if z is not None and len(z)>100:return z.assign(date=pd.to_datetime(z.date)).sort_values('date').drop_duplicates('date').set_index('date')
  except Exception: pass
D={s:load(s) for s in U};D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index();R=C.pct_change();
csmed=R.median(axis=1);res=R.sub(csmed,axis=0);vol=R.rolling(20,min_periods=15).std();trend=R.rolling(40,min_periods=30).sum()
f=-(res.rolling(5,min_periods=4).sum()/vol.rolling(20,min_periods=15).mean())
f=f*(1-0.5*np.sign(trend)*np.sign(R.rolling(10,min_periods=8).sum()));f=f.replace([np.inf,-np.inf],np.nan).shift(1)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],C.pct_change(10).shift(-10).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'ic_dates',len(o),'avg_n',round(o.n.mean(),3),'coverage',round(o.n.mean()/15,4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a+'-'+b,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
for h in [1,3,5,10,20]:
 yy=C.pct_change(h).shift(-h);z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('decay',h,round(float(np.nanmean(z)),6),len(z))
print('rank_turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),4))
f.to_csv('scripts/miner_1_20321125_residual_regime_reversal_signal.csv',index_label='date')
