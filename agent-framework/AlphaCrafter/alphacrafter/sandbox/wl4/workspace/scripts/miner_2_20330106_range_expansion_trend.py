import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=3400)
            if x is not None and len(x): return x.set_index('date')
        except Exception: pass
    return None
def main():
    raw={s:load(s) for s in SYMS}; raw={s:x for s,x in raw.items() if x is not None}
    p=pd.concat({s:x['close'].astype(float) for s,x in raw.items()},axis=1).sort_index(); r=p.pct_change()
    # Continuation after a lagged medium trend, emphasizing genuine recent range expansion.
    ret20=p.pct_change(20).shift(1)
    vol10=r.rolling(10,min_periods=8).std().shift(1)
    vol60=r.rolling(60,min_periods=40).std().shift(1)
    expansion=(vol10/(vol60+1e-8)-1).clip(-3,3)
    persistence=(r.gt(0).rolling(20,min_periods=15).mean().shift(1)-.5)*2
    fac=(ret20/(vol60*np.sqrt(20)+1e-8)* (1+0.5*expansion) * (0.5+0.5*persistence)).clip(-8,8)
    fac=fac.sub(fac.mean(axis=1),axis=0)
    for h in (5,10,20,30):
        fw=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
        for dt in p.index:
            z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
            if len(z)>=8:
                vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
        ic=pd.Series(vals,index=dates).dropna(); ir=ic.mean()/ic.std(ddof=1)*np.sqrt(252)
        print('H',h,'dates',len(ic),'avgN',round(np.mean(ns),2),'minN',min(ns),'IC',round(ic.mean(),6),'ICIR',round(ir,4),'hit',round((ic>0).mean(),4))
        print('recent',[(n,round(ic.tail(n).mean(),6),round(ic.tail(n).mean()/ic.tail(n).std(ddof=1)*np.sqrt(252),4)) for n in (260,520,780) if len(ic)>=n])
    ranks=fac.rank(axis=1,pct=True)
    print('symbols',len(raw),'coverage',round((fac.notna().sum(axis=1)/len(raw)).mean(),4),'turnover',round((ranks-ranks.shift(1)).abs().mean(axis=1).mean(),4),'cutoff',p.index.max())
if __name__=='__main__': main()
