import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=4000)
  except:pass
  if d is not None and len(d)>200:break
 if d is not None: px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); disp=r.rolling(20).std().mean(axis=1)
# cross-sectional dispersion standardized against its trailing history, lagged to avoid lookahead
state=(disp/disp.rolling(120).median()).shift(1).clip(0.5,2.0)
f=(-(p.pct_change(5)/r.rolling(20).std().replace(0,np.nan))).shift(1).mul(state,axis=0)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=np.array([x[1] for x in rows]); n=len(a); c=n//3
print('factor=dispersion_scaled_volnorm_reversal dates',n,'avg_n',np.mean([x[2] for x in rows]))
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.5f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),np.mean([x[2] for x in rows])/len(U),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
print('regimes',[(np.mean(q),np.mean(q)/np.std(q,ddof=1)) for q in (a[:c],a[c:2*c],a[2*c:])])
for h in [1,5,10,20,40]:
 aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8:aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(float(np.mean(aa)),6),len(aa))
f.index=f.index.strftime('%Y-%m-%d'); f.to_csv('scripts/miner_1_20300729_disp_scaled_reversal_signal.csv'); pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20300729_disp_scaled_reversal_ic.csv',index=False)
