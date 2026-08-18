import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=3400)
            if x is not None and len(x): return x.set_index('date')['close'].astype(float)
        except Exception: pass
    return None
def main():
    d={s:load(s) for s in SYMS}; d={s:x for s,x in d.items() if x is not None}
    p=pd.concat(d,axis=1).sort_index(); r=p.pct_change()
    mom=p.pct_change(30).shift(1)
    down=r.where(r<0,0).rolling(60,min_periods=30).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True).shift(1)*np.sqrt(30)
    consistency=(r.gt(0).rolling(60,min_periods=30).mean().shift(1)-0.5)*2
    fac=(mom/(down+1e-8)*consistency).clip(-6,6)
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
    print('symbols',len(d),'coverage',round((fac.notna().sum(axis=1)/len(d)).mean(),4),'turnover',round((ranks-ranks.shift(1)).abs().mean(axis=1).mean(),4),'cutoff',p.index.max())
    fac.tail(1).T.rename(columns={fac.index[-1]:'signal'}).to_csv('scripts/miner_2_20321223_asymmetric_trend_quality_signal.csv')
if __name__=='__main__': main()
