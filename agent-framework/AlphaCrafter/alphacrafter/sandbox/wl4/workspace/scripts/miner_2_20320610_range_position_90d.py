import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s,days=3200)
            if d is not None and len(d): return d.set_index('date')['close'].astype(float)
        except Exception: pass
    return None
def calc(p):
    q=p.shift(1); hi=q.rolling(90,min_periods=65).max(); lo=q.rolling(90,min_periods=65).min()
    pos=(q-lo)/(hi-lo).replace(0,np.nan)
    vol=p.pct_change().shift(1).rolling(40,min_periods=30).std()*np.sqrt(252)
    return (0.5-pos)/vol.clip(lower=0.05)
def main():
    px={s:load(s) for s in SYMS}; px={s:x for s,x in px.items() if x is not None}
    p=pd.DataFrame(px).sort_index(); fac=calc(p)
    out=[]
    for h in (5,10,20):
        fwd=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
        for dt in p.index:
            z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
            if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ds.append(dt); ns.append(len(z))
        ic=pd.Series(vals,index=ds).replace([np.inf,-np.inf],np.nan).dropna()
        print('H',h,'dates',len(ic),'avgN',round(np.mean(ns),2),'minN',min(ns),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1)*np.sqrt(252),4),'hit',round((ic>0).mean(),4))
        if h==10:
            recent=[]
            for n in (260,520,780):
                x=ic.tail(n); recent.append((n,round(x.mean(),6),round(x.mean()/x.std(ddof=1)*np.sqrt(252),4)))
            print('recent',recent)
    valid=fac.notna().sum(axis=1); print('symbols',len(px),'coverage',round((valid/len(px)).mean(),4),'cutoff',p.index.max())
    # deterministic artifact for provenance
    fac.tail(1).T.rename(columns={fac.index[-1]:'signal'}).to_csv('scripts/miner_2_20320610_range_position_90d_signal.csv')
if __name__=='__main__': main()
