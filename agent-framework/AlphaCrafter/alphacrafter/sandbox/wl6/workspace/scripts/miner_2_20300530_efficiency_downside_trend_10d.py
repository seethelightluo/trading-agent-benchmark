import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def main():
    sig, fwd, all_dates = {}, {}, set()
    for s in U:
        d=get_stock_daily_data(s,days=1800)
        if d is None or len(d)==0: d=get_index_daily_data(s,days=1800)
        if d is None or len(d)<80: continue
        d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
        r=d.close.pct_change()
        # Trend quality: directional efficiency times downside-risk-adjusted return.
        # Efficiency suppresses choppy paths; downside deviation penalizes fragile trends.
        path=r.abs().rolling(20,min_periods=20).sum()
        efficiency=(r.rolling(20,min_periods=20).sum().abs()/(path+1e-12))
        downside=np.sqrt((r.clip(upper=0)**2).rolling(20,min_periods=20).mean())+1e-8
        direction=np.sign(r.rolling(20,min_periods=20).sum())
        sig[s]=direction*efficiency*(r.rolling(20,min_periods=20).sum()/downside)
        fwd[s]={h:d.close.shift(-h)/d.close-1 for h in [5,10,20]}
        all_dates.update(d.index)
    for h in [5,10,20]:
        ics=[]; ns=[]
        for t in sorted(all_dates):
            a=[];b=[]
            for s in sig:
                if t in sig[s].index and np.isfinite(sig[s].loc[t]) and np.isfinite(fwd[s][h].loc[t]): a.append(sig[s].loc[t]);b.append(fwd[s][h].loc[t])
            if len(a)>=8 and np.std(a)>0 and np.std(b)>0:
                ics.append(spearmanr(a,b).statistic);ns.append(len(a))
        x=np.asarray(ics); print(f'horizon={h} valid_dates={len(x)} avg_instruments={np.mean(ns):.3f} IC={x.mean():.8f} ICIR={x.mean()/(x.std(ddof=1)+1e-12):.8f} hit={np.mean(x>0):.6f} coverage={np.mean(ns)/15:.6f}')
if __name__=='__main__': main()
