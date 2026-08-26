import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=4000)
  except:pass
  if d is not None and len(d)>200:break
 if d is not None:px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change();
# low volatility with medium-term volatility improving (declining risk), lagged
v=r.rolling(20).std(); f=(-(v)+v.rolling(10).mean()-v.rolling(40).mean()).shift(1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=np.array([x[1] for x in rows]); n=len(a);c=n//3
print('factor=lagged_lowvol_improving dates',n,'avg_n',np.mean([x[2] for x in rows]));print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.5f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),np.mean([x[2] for x in rows])/len(U),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()));print('regimes',[(np.mean(q),np.mean(q)/np.std(q,ddof=1)) for q in(a[:c],a[c:2*c],a[2*c:])]);
for h in [1,5,10,20,40]:
 aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8:aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(float(np.mean(aa)),6),len(aa))
