import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2031-10-15')
# align close panel, retaining all observed sessions
series=[]
for a in assets:
    x=pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date'])
    x=x[x.date<=cutoff].set_index('date').close.rename(a)
    series.append(x)
close=pd.concat(series,axis=1).sort_index()
r=np.log(close).diff()
# Factor: inverse peer up/down co-movement asymmetry, 60 observations
# high raw down-correlation relative to up-correlation was empirically positive.
sig=pd.DataFrame(np.nan,index=close.index,columns=assets)
for t in range(60,len(close)):
    w=r.iloc[t-60:t]
    med=w.median(axis=1)
    down=med<0; up=med>0
    if down.sum()<12 or up.sum()<12: continue
    for a in assets:
        y=w[a]; peer=w.drop(columns=a).mean(axis=1)
        d=pd.concat([y[down],peer[down]],axis=1).dropna()
        u=pd.concat([y[up],peer[up]],axis=1).dropna()
        if len(d)>=12 and len(u)>=12:
            sig.loc[sig.index[t],a]=d.iloc[:,0].corr(d.iloc[:,1])-u.iloc[:,0].corr(u.iloc[:,1])

def ic_stats(h, mask=None):
    fr=np.log(close.shift(-h)/close)
    vals=[]; ns=[]
    for dt in close.index:
        if mask is not None and not mask.loc[dt]: continue
        z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
    x=np.array(vals); mean=x.mean(); sd=x.std(ddof=1)
    return dict(ic=float(mean),icir=float(mean/sd) if sd else np.nan,hit=float((x>0).mean()),dates=len(x),names=float(np.mean(ns)),se=float(sd/np.sqrt(len(x))) if len(x)>1 else np.nan)
print('FACTOR inverse_peer_up_down_comovement_asymmetry_60obs')
print('cutoff',cutoff.date(),'panel_dates',len(close),'assets',len(assets),'signal_coverage',float(sig.notna().mean().mean()),'mean_available',float(sig.notna().sum(axis=1).mean()))
for h in [1,5,10,20]: print('H',h,ic_stats(h))
# selected 10d calendar regimes
for label,lo,hi in [('2020-2021','2020-01-01','2021-12-31'),('2022-2023','2022-01-01','2023-12-31'),('2024-2026','2024-01-01','2026-12-31'),('2027-2030','2027-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-10-15')]:
 m=pd.Series((close.index>=lo)&(close.index<=hi),index=close.index)
 print('REGIME',label,ic_stats(10,m))
# rank stability on adjacent signal dates
cs=[]
for i in range(1,len(sig)):
 z=pd.concat([sig.iloc[i-1],sig.iloc[i]],axis=1).dropna()
 if len(z)>=8: cs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('rank_stability_1d',float(np.mean(cs)),'turnover',float(1-np.mean(cs)),'stability_dates',len(cs))
# recent (post prior validation) 10-day diagnostic
m=pd.Series(close.index>=pd.Timestamp('2031-05-15'),index=close.index)
print('POST_PRIOR_10D',ic_stats(10,m))
# save artifact for reproducibility
sig.to_pickle('scripts/miner_2_20311016_inverse_peer_up_down_comovement_60obs_revalidation_signal.pkl')
