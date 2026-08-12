import os, json
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index()
# Downside-persistence trend: positive 40d return, penalized by downside deviation and recent drawdown.
ret=px.pct_change()
r40=px.pct_change(40)
down=ret.where(ret<0,0).rolling(40,min_periods=25).std()
# trend persistence = fraction positive days; modestly rewards consistent path
persist=(ret>0).rolling(40,min_periods=25).mean()
raw=(r40/(down*np.sqrt(40)+1e-12))* (0.5+persist)
sig=raw.rank(axis=1,pct=True).rolling(5,min_periods=3).mean().shift(1)
# forward compounded close returns
out=[]
for h in [1,5,10,20]:
    fwd=px.shift(-h)/px-1
    vals=[]; ninst=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ninst.append(len(z))
    a=np.asarray(vals); ic=float(np.nanmean(a)); sd=float(np.nanstd(a,ddof=1)); icir=ic/sd*np.sqrt(252) if sd>0 else 0
    print(f'H={h} dates={len(a)} avg_inst={np.mean(ninst):.2f} IC={ic:.6f} ICIR={icir:.6f} hit={np.mean(a>0):.4f}')
# diagnostics at admission horizon
h=20; fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
a=np.asarray(vals)
print('coverage',float(sig.notna().sum().sum()/sig.size),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean().mean()),'start',dates[0],'end',dates[-1])
# persist signal artifact
os.makedirs('scripts',exist_ok=True)
sig.reset_index().to_csv('scripts/miner_2_20301031_downside_persistence_trend_signal.csv',index=False)
print('artifact rows',len(sig),'assets',len(sig.columns))
