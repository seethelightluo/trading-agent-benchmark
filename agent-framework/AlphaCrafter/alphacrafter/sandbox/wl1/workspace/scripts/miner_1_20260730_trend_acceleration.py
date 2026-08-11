import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data

SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def main():
    px={}
    for s in SYMS:
        d=get_index_daily_data(s, days=1800)
        if d is not None and len(d):
            x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); px[s]=x.set_index('date').close
    close=pd.DataFrame(px).sort_index().ffill()
    r5=close.pct_change(5); r20=close.pct_change(20)
    # trend acceleration: recent 5d return versus average weekly pace over prior 20d
    f=r5-r20/4
    f=f.replace([np.inf,-np.inf],np.nan)
    fw=close.pct_change().shift(-1)
    ics=[]; cov=[]; dates=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/len(SYMS)); dates.append(dt)
    a=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
    mean=a.mean(); sd=a.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
    # turnover of cross-sectional ranks, adjacent valid dates
    ranks=f.rank(axis=1,pct=True); turns=[]
    for i in range(1,len(ranks)):
        q=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
        if len(q)>=8: turns.append(np.mean(np.abs(q.iloc[:,1]-q.iloc[:,0])))
    print({'dates':len(a),'instruments':len(SYMS),'start':str(a.index.min().date()),'end':str(a.index.max().date()),'IC':mean,'ICIR':icir,'hit':float((a>0).mean()),'coverage':float(np.mean(cov)),'rank_turnover':float(np.mean(turns)),'decay_5d':float(pd.concat([f,close.pct_change(5).shift(-5)],axis=1).dropna().groupby(level=0).apply(lambda x: np.nan).mean()) if False else 'see below'})
    for h in [1,5,10]:
        rr=close.pct_change(h).shift(-h); vals=[]
        for dt in f.index:
            z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
            if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
        print('decay',h,float(pd.Series(vals).dropna().mean()),len(vals))
if __name__=='__main__': main()
