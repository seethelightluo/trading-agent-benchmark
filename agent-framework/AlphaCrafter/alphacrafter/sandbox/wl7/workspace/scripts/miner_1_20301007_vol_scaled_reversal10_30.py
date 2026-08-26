import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: d=fn(s,days=4000)
  except Exception: pass
  if d is not None and len(d)>200: break
 if d is not None:
  q=d.copy(); q['date']=pd.to_datetime(q.date); px[s]=q.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Medium-horizon reversal, scaled by 30-session realized risk; one-session lag prevents lookahead.
vol=r.rolling(30,min_periods=20).std()
f=(-(r.rolling(10,min_periods=10).sum()/vol)).shift(1)
y=p.pct_change(10).shift(-10)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=np.array([x[1] for x in rows]); ns=np.array([x[2] for x in rows])
print('factor=vol_scaled_reversal10_30 dates',len(a),'avg_n',ns.mean(),'assets',len(U))
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),ns.mean()/len(U)))
c=len(a)//3; print('regimes',[(q.mean(),q.mean()/q.std(ddof=1),len(q)) for q in (a[:c],a[c:2*c],a[2*c:])])
for h in [1,5,10,20,40]:
 aa=[]; yy=p.pct_change(h).shift(-h)
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(float(np.mean(aa)),6),len(aa))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20301007_vol_scaled_reversal10_30_signal.csv')
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20301007_vol_scaled_reversal10_30_ic.csv',index=False)
