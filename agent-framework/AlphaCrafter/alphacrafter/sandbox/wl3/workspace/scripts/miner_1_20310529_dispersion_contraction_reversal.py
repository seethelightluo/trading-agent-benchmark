import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: x=fn(s,days=5000)
        except Exception: x=None
        if x is not None and len(x): break
    if x is not None and len(x):
        x=x.copy(); x.date=pd.to_datetime(x.date)
        D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
r3=p.pct_change(3); disp=r.std(axis=1)
# Reversal after an elevated dispersion shock that is now contracting.
# Signal is lagged: values at date t predict returns from t+1 onward.
elev=disp > disp.rolling(252,min_periods=126).median()
contract=(disp < disp.shift(1)) & elev.shift(1)
res=r3.sub(r3.median(axis=1),axis=0)
f=-res/(r.rolling(20).std()*np.sqrt(3)+1e-12)*contract.astype(float).values[:,None]
rows_by={}
for h in [1,3,5,6,10]:
 rows=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rows.append((p.index[i],z.f.corr(z.y),len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); rows_by[h]=a
 print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),3),'coverage',round(a.n.sum()/(len(a)*15),4),'IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean()))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
  q=a.loc[lo:hi]
  if len(q): print(lo+'-'+hi,len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12)))
rank=f.rank(axis=1,pct=True); print('turnover',((rank.diff().abs().mean(axis=1)/2).mean()),'signal_coverage',f.notna().mean().mean(),'last_date',p.index[-1].date())
f.index.name='date'; f.reset_index().to_csv('scripts/miner_1_20310529_dispersion_contraction_reversal_signal.csv',index=False)
