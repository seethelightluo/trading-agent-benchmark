import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index(); frames[s]=d['close'].astype(float)
p=pd.concat(frames,axis=1).sort_index(); r=p.pct_change()
mom=p.pct_change(20).shift(1); breadth=(mom>0).mean(axis=1); reg=(breadth-0.5)*2
sig=mom.mul(reg,axis=0).clip(-0.5,0.5)
for h in [1,3,5,10]:
    fwd=p.shift(-h)/p-1; vals=[]; nms=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
        if ok.sum()>=8: vals.append(x[ok].corr(y[ok])); nms.append(ok.sum())
    a=np.array(vals,float); ic=np.nanmean(a); sd=np.nanstd(a,ddof=1)
    print('H',h,'dates',len(a),'avgN',round(float(np.mean(nms)),2),'IC',round(float(ic),6),'ICIR',round(float(ic/sd),6),'hit',round(float(np.mean(a>0)),4))
    if h==10:
        for window in [180,360]:
            z=a[-window:]; print('RECENT',window,'IC',round(float(np.nanmean(z)),6),'ICIR',round(float(np.nanmean(z)/np.nanstd(z,ddof=1)),6),'dates',len(z))
print('coverage',float(sig.notna().sum().sum()/(len(sig)*len(U))),'dates',len(sig),'instruments',len(U),'breadth mean',float(breadth.mean()))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20290111_breadth_conditioned_trend_signal.csv',index=False)
