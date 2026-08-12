import numpy as np,pandas as pd
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
# Volatility-managed medium momentum: 15-day return divided by downside deviation over 30 days.
# This rewards persistent gains while penalizing assets with frequent negative shocks.
down=r.clip(upper=0).rolling(30,min_periods=20).std()
f=p.pct_change(15)/(down*np.sqrt(15)+1e-12)
# one-day lag makes the signal observable before the forward interval
f=f.shift(1)
rows=[]
for i in range(len(p)-11):
    z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+11]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('shape',p.shape,'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15),'last_date',p.index[-1].date())
print('H10 IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q): print(lo+'-'+hi,'dates',len(q),'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12),(q.ic>0).mean()))
for h in [1,3,5,10,20]:
 rr=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h+1]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('H',h,'dates',len(rr),'IC %.8f ICIR %.8f'%(np.nanmean(rr),np.nanmean(rr)/(np.nanstd(rr,ddof=1)+1e-12)))
print('turnover', (f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean())
f.index.name='date'; f.reset_index().to_csv('scripts/miner_3_20310626_downside_momentum_signal.csv',index=False)
