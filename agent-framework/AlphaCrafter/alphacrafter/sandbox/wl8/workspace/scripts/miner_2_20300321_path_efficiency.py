import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=get_stock_daily_data(s,4000)
    if d is None or len(d)<150: d=get_index_daily_data(s,4000)
    if d is None: continue
    d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d['r']=d.close.pct_change()
    # signed path efficiency: directional displacement / total traveled path, with volatility normalization
    net=d.close/d.close.shift(60)-1
    path=d.r.abs().rolling(60).sum()
    vol=d.r.rolling(20).std()*np.sqrt(20)
    # reward persistent directional movement and penalize unstable path
    d['factor']=(net/path).clip(-1,1) / vol.replace(0,np.nan)
    d['symbol']=s
    rows.append(d[['date','symbol','factor','close']])
x=pd.concat(rows).sort_values(['symbol','date'])
x['fwd5']=x.groupby('symbol').close.shift(-5)/x.close-1
x['fwd10']=x.groupby('symbol').close.shift(-10)/x.close-1
x['fwd20']=x.groupby('symbol').close.shift(-20)/x.close-1
# lag signal one day to avoid same close
x['factor']=x.groupby('symbol').factor.shift(1)

def calc(h):
    z=x.dropna(subset=['factor','fwd'+str(h)])
    obs=[]
    for dt,g in z.groupby('date'):
        if len(g)>=8: obs.append((dt,g.factor.corr(g['fwd'+str(h)]),len(g)))
    a=pd.DataFrame(obs,columns=['date','ic','n']).dropna()
    ic=a.ic.mean(); sd=a.ic.std(ddof=1)
    return len(a),a.n.mean(),ic,ic/sd*np.sqrt(1) if sd else np.nan,(a.ic>0).mean(),a.n.sum()/len(a)/15
for h in [5,10,20]: print(h,calc(h))
# regime / trailing
z=x.dropna(subset=['factor','fwd10']); obs=[]
for dt,g in z.groupby('date'):
 if len(g)>=8: obs.append((dt,g.factor.corr(g.fwd10)))
a=pd.DataFrame(obs,columns=['date','ic']).dropna(); a.date=pd.to_datetime(a.date)
for name,lo,hi in [('2020-2023','2020','2023-12-31'),('2024-2026','2024','2026-07-15'),('online1','2026-07-16','2028-12-31'),('recent','2029-01-01','2030-03-20'),('trail180','2029-07-01','2030-03-20')]:
 q=a[(a.date>=lo)&(a.date<=hi)]; print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
x.to_csv('scripts/miner_2_20300321_path_efficiency_signal.csv',index=False)
print('rows',len(x),'symbols',x.symbol.nunique(),'dates',x.date.nunique())
