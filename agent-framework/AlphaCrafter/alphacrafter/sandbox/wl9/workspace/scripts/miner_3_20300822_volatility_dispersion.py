import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Volatility-dispersion factor: relative idiosyncratic volatility (asset 20d vol / cross-sectional median), inverted.
data={}
for s in U:
    d=get_stock_daily_data(s, days=3000)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
        data[s]=d.close.astype(float)
p=pd.DataFrame(data).sort_index(); r=p.pct_change()
vol=r.rolling(20,min_periods=15).std(); med=vol.median(axis=1)
f=-(vol.div(med,axis=0)-1.0) # low vol ranks high
# lag signal one completed session
f=f.shift(1)
res=[]
for h in [10,20,40,60]:
    fr=p.shift(-h).div(p)-1
    vals=[]; dates=[]; ns=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
    a=pd.Series(vals,index=dates).dropna(); ic=a.mean(); sd=a.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
    print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.6f} ICIR={icir:.6f} hit={np.mean(a>0):.4f}')
    if h==40:
      for lo,hi,nm in [('2024-01-01','2026-12-31','2024-26'),('2027-01-01','2029-12-31','2027-29'),('2030-01-01','2030-12-31','2030YTD')]:
        q=a[(a.index>=lo)&(a.index<=hi)]; print(f'  regime={nm} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan:.6f}')
f.to_csv('scripts/miner_3_20300822_volatility_dispersion_signal.csv', index_label='date')
# turnover rank proxy
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean(); print(f'turnover_proxy={turn:.6f} instruments={len(data)} dates={len(p)}')
