import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=4000)
  except Exception: pass
  if d is not None and len(d)>200: break
 if d is not None:
  d=d.copy(); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); e=r.sub(r.mean(axis=1),axis=0)
down=e.clip(upper=0).fillna(0).pow(2).rolling(40,min_periods=30).mean().pow(.5).replace(0,np.nan)
f=-(e.rolling(20,min_periods=20).sum()/down).shift(1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=np.array([x[1] for x in rows]); ns=[x[2] for x in rows]
print('factor=peer_residual_downside_reversal20 dates',len(a),'avg_n',np.mean(ns),'assets',len(U))
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),np.mean(ns)/len(U)))
c=len(a)//3; qs=(a[:c],a[c:2*c],a[2*c:]); print('regimes',[(np.mean(q),np.mean(q)/np.std(q,ddof=1)) for q in qs])
for h in [1,5,10,20,40]:
 aa=[]; y=p.pct_change(h).shift(-h)
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(float(np.mean(aa)),6),len(aa))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20300826_peer_residual_downside_reversal20_signal.csv'); pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20300826_peer_residual_downside_reversal20_ic.csv',index=False)
