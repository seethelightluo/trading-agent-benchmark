import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s, days=3200)
            if x is not None and len(x): return x
        except Exception: pass
    return None

def main():
    px={}
    for s in SYMS:
        d=get(s)
        if d is not None: px[s]=d.set_index('date')['close'].astype(float)
    p=pd.DataFrame(px).sort_index()
    # lagged 60d normalized range position, with volatility scaling; low position is expected to rebound
    ret=p.shift(1)/p.shift(61)-1
    hi=p.shift(1).rolling(60,min_periods=45).max(); lo=p.shift(1).rolling(60,min_periods=45).min()
    pos=(p.shift(1)-lo)/(hi-lo)
    vol=p.pct_change().shift(1).rolling(40,min_periods=30).std()*np.sqrt(252)
    # centered range position and inverse vol, raw high means low position / attractive
    fac=(0.5-pos)/(vol.clip(lower=0.05))
    # forward 10 trading day return
    fwd=p.shift(-10)/p-1
    rows=[]; dates=[]
    for dt in p.index:
        a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
    ic=pd.Series(rows,index=dates)
    print('symbols',len(px),'dates',len(ic),'avgN',np.mean([pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna().shape[0] for d in dates]))
    print('cutoff',p.index.max(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit', (ic>0).mean(),'minN',min(pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna().shape[0] for d in dates))
    for n in (260,520,780):
        q=ic.tail(n); print('recent',n,q.mean(),q.mean()/q.std(ddof=1),len(q))
    # decay
    for h in (5,10,20):
        ff=p.shift(-h)/p-1; rr=[]
        for dt in p.index:
            z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
            if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
        q=pd.Series(rr).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
    print('coverage',fac.notna().sum().sum()/fac.size)
    # signal artifact for audit
    out=fac.tail(900).reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
    out.to_csv('scripts/miner_2_20320527_range_position_reversal_signal.csv',index=False)
if __name__=='__main__': main()
