import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    try: d=get_index_daily_data(s,days=4100)
    except Exception: d=get_stock_daily_data(s,days=4100)
    if d is not None and len(d):
        d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float).sort_index()
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff()
# Downside-risk-adjusted medium momentum, with a slow trend confirmation.
down=r.clip(upper=0).rolling(40).std()*np.sqrt(40)
raw=(p/p.shift(20)-1)/(down+1e-8) + .25*(p/p.shift(60)-1)/(r.rolling(60).std()*np.sqrt(60)+1e-8)
f=raw.shift(1).rank(axis=1,pct=True)
print('instruments',len(px),'date_range',p.index.min(),p.index.max())
def calc(h):
 y=p.shift(-h)/p-1; vals=[]; ds=[]; cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ds.append(dt); cov.append(len(z)/len(U))
 x=pd.Series(vals,index=ds).dropna(); return x
ics=calc(1)
print('dates',len(ics),'avg_names',p.notna().sum(axis=1).mean(),'coverage',np.mean([len(pd.concat([f.loc[d],p.shift(-1).loc[d]],axis=1).dropna())/15 for d in ics.index]))
for h in [1,5,10,20]:
 x=calc(h); print('%dd IC %.6f ICIR %.6f hit %.4f'%(h,x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
print('turnover',f.diff().abs().mean(axis=1).dropna().mean())
for a,g in ics.groupby(ics.index.year): print(a,'IC %.6f n=%d'%(g.mean(),len(g)))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20310320_downside_momentum_signal.csv',index=False)
