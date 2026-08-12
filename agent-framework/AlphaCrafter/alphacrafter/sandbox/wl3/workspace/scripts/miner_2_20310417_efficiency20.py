import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: x=fn(s,days=5000)
        except Exception: x=None
        if x is not None and len(x): break
    if x is not None and len(x):
        x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Efficiency: realized 20-session return per unit of recent absolute-return activity, volatility normalized.
f=((p/p.shift(20)-1)/(r.abs().rolling(20).sum()+1e-12))/(r.rolling(40).std()*np.sqrt(20)+1e-12)
rows=[]; h=20
for i in range(len(p)-h):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.std()>0 and z.y.std()>0: rows.append((p.index[i],z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic.mean(); icir=ic/(a.ic.std(ddof=1)+1e-12)
print('factor=efficiency_20d horizon=20 validation_end',p.index[-1].date()); print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15)); print('IC %.8f ICIR %.8f hit %.4f'%(ic,icir,(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q): print(lo+'-'+hi,'dates',len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12)))
rank=f.rank(axis=1,pct=True); print('turnover',float((rank.diff().abs().mean(axis=1)/2).mean()))
f.index.name='date'; f.reset_index().to_csv('scripts/miner_2_20310417_efficiency20_signal.csv',index=False)
