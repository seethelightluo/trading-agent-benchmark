import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,5000)
    if d is None or len(d)==0: d=get_index_daily_data(s,5000)
    if d is not None: frames[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(frames).sort_index().loc['2020-01-01':'2034-09-14']
# candidate: lagged 5d cross-sectional residual reversal, normalized by 20d vol
r=p.pct_change(); f=-(r.rolling(5).sum().shift(1)).div(r.rolling(20).std().shift(1))
# cross-sectional demean (ranking equivalent, improves comparability)
f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for h in [1,5,10,20]:
    fr=p.pct_change(h).shift(-h)
    ics=[]; n=[]; turns=[]
    dates=sorted(set(f.index)&set(fr.index))
    prev=None
    for dt in dates:
        x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            ics.append(z.iloc[:,0].corr(z.iloc[:,1])); n.append(len(z))
            sig=x.reindex(U).rank(pct=True)
            if prev is not None: turns.append(np.nanmean(abs(sig-prev)))
            prev=sig
    a=np.array(ics,dtype=float); a=a[np.isfinite(a)]
    ic=a.mean(); sd=a.std(ddof=1); ir=ic/sd*np.sqrt(252) if sd else np.nan
    # regime split
    def seg(a0,a1):
      q=[v for dt,v in zip(dates,ics) if a0<=str(dt.date())<=a1 and np.isfinite(v)]
      return (float(np.mean(q)),len(q)) if q else (np.nan,0)
    print('horizon',h,'dates',len(a),'avgN',round(np.mean(n),2),'coverage',round(len(a)/(len(dates) or 1),4),'IC',round(ic,6),'ICIR',round(ir,6),'turnover',round(float(np.nanmean(turns)),4),'regimes',seg('2020-01-01','2026-07-15'),seg('2026-07-16','2034-09-14'))
# save signal artifact
out=f.copy(); out.to_csv('../persistent/miner_2_20340915_short_reversal_volnorm_signal.csv')
print('shape',p.shape,'assets',len(frames),'artifact written')
