import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is not None and len(d)>100: frames[s]=d[['date','close']].copy()
rows=[]
for s,d in frames.items():
    d=d.sort_values('date').set_index('date'); r=np.log(d.close).diff()
    f=1/(r.rolling(20).std().shift(1)+1e-8); fr=np.log(d.close.shift(-10)/d.close)
    rows.append(pd.DataFrame({'f':f,'y':fr,'s':s}).dropna())
x=pd.concat(rows); ics=[]
for dt,g in x.groupby(level=0):
    if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ics.append((dt,g.f.corr(g.y),len(g)))
ic=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
def met(q):
 a=ic.loc[ic.index>=q]; return len(a),a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),a.n.mean()
print('instruments',len(frames),'dates',len(ic),'avg_n',ic.n.mean())
for name,q in [('full','2026-07-16'),('2028+','2028-01-01'),('2029','2029-01-01'),('recent180','2029-09-01')]: print(name,met(q))
sig=x.reset_index().rename(columns={'date':'asof_date','f':'signal'}); sig.to_csv('scripts/miner_1_20300307_lowvol_signal.csv',index=False); ic.reset_index().to_csv('scripts/miner_1_20300307_lowvol_ic.csv',index=False)
print('cells',len(x),'coverage',len(x)/(len(frames)*len(set(x.index))))
