import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={s:get_stock_daily_data(s,days=5000) for s in U}
px=pd.concat({s:d.set_index('date').close for s,d in xs.items() if d is not None},axis=1).sort_index().ffill()
ret=px.pct_change(); bench=ret.mean(axis=1)
# Leadership-confirmed rebound: recent 5d drawdown, scaled by vol, only where medium-horizon up-day breadth is positive vs benchmark.
rows=[]
for t in range(70,len(px)-10):
    date=px.index[t]
    r5=px.iloc[t]/px.iloc[t-5]-1
    vol=ret.iloc[t-20:t].std()*np.sqrt(20)
    breadth=ret.iloc[t-60:t].gt(0).mean()
    bbench=ret.iloc[t-60:t].mean(axis=1).gt(0).mean() # unused benchmark temporal breadth
    # cross-sectional leadership: instrument breadth minus contemporaneous cross-asset average breadth
    csbase=breadth.mean()
    f=-(r5/vol)*(breadth-csbase)
    f=f.replace([np.inf,-np.inf],np.nan).dropna()
    fr=px.iloc[t+10]/px.iloc[t]-1
    z=pd.concat([f.rename('f'),fr.rename('r')],axis=1).dropna()
    if len(z)>=8:
        rows.append((date,z.f.corr(z.r),len(z),z.f.rank().corr(z.r.rank())))
out=pd.DataFrame(rows,columns=['date','ic','n','rankic']).set_index('date')
# signal artifact for latest historical date
sig=[]
t= len(px)-11
r5=px.iloc[t]/px.iloc[t-5]-1; vol=ret.iloc[t-20:t].std()*np.sqrt(20); breadth=ret.iloc[t-60:t].gt(0).mean(); f=(-(r5/vol)*(breadth-breadth.mean())).replace([np.inf,-np.inf],np.nan)
for s,v in f.items(): sig.append({'date':px.index[t].strftime('%Y-%m-%d'),'symbol':s,'signal':float(v) if pd.notna(v) else np.nan})
pd.DataFrame(sig).to_csv('scripts/miner_3_20320902_leadership_rebound_signal.csv',index=False)
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=out.loc[(out.index>=lo)&(out.index<=hi)]; print(lo+'-'+hi,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std() if q.ic.std()>0 else np.nan,'hit',(q.ic>0).mean())
print('ALL dates',len(out),'avg n',out.n.mean(),'IC',out.ic.mean(),'ICIR',out.ic.mean()/out.ic.std(),'hit',(out.ic>0).mean(),'rankIC',out.rankic.mean())
print('coverage',len(f.dropna())/len(U),'turnover not applicable rank signal')
