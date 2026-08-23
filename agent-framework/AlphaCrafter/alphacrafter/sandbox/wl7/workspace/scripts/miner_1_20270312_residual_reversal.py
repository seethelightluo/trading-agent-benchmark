import os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-03-12')
rows=[]
prices={}
for a in ASSETS:
    p='../persistent/stock_data/'+a+'.csv'
    if not os.path.exists(p): continue
    d=pd.read_csv(p,parse_dates=['date']).sort_values('date')
    d=d[d.date<=END].set_index('date')
    prices[a]=d.close
px=pd.DataFrame(prices).sort_index()
ret=px.pct_change()
# Cross-asset residual: asset 3d return minus contemporaneous cross-sectional median 3d return,
# scaled by trailing 20d idiosyncratic volatility; all signal inputs end at t, target is t+1.
r3=px.pct_change(3)
market=r3.median(axis=1)
resid=r3.sub(market,axis=0)
idio=(ret.sub(ret.median(axis=1),axis=0)).rolling(20,min_periods=12).std()
sig=-resid/idio.replace(0,np.nan)
# archive signal values by date/assets for deterministic audit
out=sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'})
out.to_csv('scripts/miner_1_20270312_residual_reversal_signal.csv',index=False)
for h in [1,5,10]:
    fwd=px.pct_change(h).shift(-h)
    ics=[]; ninst=[]; dates=[]
    for dt in sig.index:
        z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
            if np.isfinite(q): ics.append(q); ninst.append(len(z)); dates.append(dt)
    x=np.array(ics)
    print('horizon',h,'dates',len(x),'avg_instruments',round(np.mean(ninst),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'coverage',round(np.mean([len(sig.loc[d].dropna())/len(ASSETS) for d in dates]),4))
# turnover rank proxy on dates with valid breadth
ranks=sig.rank(axis=1,pct=True); common=ranks.dropna(how='all').diff().abs().mean(axis=1).dropna()
print('turnover_proxy',round(common.mean(),6),'period',px.index.min().date(),px.index.max().date(),'assets',len(prices))
# regime halves
valid=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],ret.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: valid.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for label, cond in [('early',lambda d:d<pd.Timestamp('2026-07-16')),('online',lambda d:d>=pd.Timestamp('2026-07-16'))]:
 x=np.array([v for d,v in valid if cond(d)])
 print(label,'dates',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
