import os, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(symbol=s,days=6000)
            if x is not None and len(x)>100:return x[['date','close']]
        except Exception: pass
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
px=pd.concat([x.set_index('date').close.rename(s) for s,x in D.items()],axis=1).sort_index()
px=px.replace([np.inf,-np.inf],np.nan).where(lambda z:z>0)
r=np.log(px).diff(); r5=np.log(px/px.shift(5)); vol=r.rolling(20,min_periods=15).std()
csmean=r5.mean(axis=1); resid=r5.sub(csmean,axis=0); disp=r.std(axis=1).rolling(5,min_periods=5).mean(); threshold=disp.rolling(120,min_periods=60).median()
f=(-resid/vol).where(disp>threshold)
fr=np.log(px.shift(-20)/px).replace([np.inf,-np.inf],np.nan)
rows=[]; sig=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  c=a[ok].corr(b[ok])
  if np.isfinite(c):rows.append((dt,c,int(ok.sum())))
 for s in a[a.notna()].index:sig.append({'date':str(dt.date()),'symbol':s,'signal':float(a[s])})
q=pd.DataFrame(rows,columns=['date','ic','n'])
print('universe',len(D),'dates',len(q),'date_range',q.date.min(),q.date.max(),'avg_n',q.n.mean())
print('IC %.8f ICIR %.8f hit %.4f recent500 %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean(),q.tail(500).ic.mean()))
coverage=np.mean([f.loc[d].notna().sum()/len(U) for d in q.date]);print('coverage %.4f'%coverage)
for lo,hi in [('2020','2024'),('2025','2029'),('2030','2034'),('2035','2035')]:
 z=q[(q.date.astype(str)>=lo)&(q.date.astype(str)<=hi)]
 if len(z):print('regime',lo,hi,len(z),'IC %.6f ICIR %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
os.makedirs('scripts',exist_ok=True);pd.DataFrame(sig).to_csv('scripts/miner_1_20350903_residual_dispersion_reversal_signal.csv',index=False);print('signal_rows',len(sig))
