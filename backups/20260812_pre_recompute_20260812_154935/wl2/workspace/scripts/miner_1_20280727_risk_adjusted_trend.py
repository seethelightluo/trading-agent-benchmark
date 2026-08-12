import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=4000)
            if x is not None and len(x)>100: return x
        except Exception: pass
    return None

def main():
    ds={s:get(s) for s in SYMS}; ds={s:x for s,x in ds.items() if x is not None}
    print('symbols',len(ds),{s:len(x) for s,x in ds.items()})
    # medium horizon risk-adjusted trend, deliberately slow turnover
    px=pd.concat({s:x.set_index('date')['close'] for s,x in ds.items()},axis=1).sort_index()
    rets=px.pct_change()
    # signal known at t: 60d return divided by downside deviation of recent 40d
    down=rets.where(rets<0).rolling(40,min_periods=20).std()
    sig=px.pct_change(60).div(down*np.sqrt(40)+1e-8)
    # winsorized cross-sectional ranks; forward non-overlapping-ish 10d return
    fwd=px.shift(-10)/px-1
    rows=[]
    for d in sig.index:
        a=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
        if len(a)>=8:
            rows.append((d,len(a),a.iloc[:,0].corr(a.iloc[:,1],method='spearman')))
    z=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
    print('obs',len(z),'avgN',z.n.mean(),'coverage',z.n.sum()/len(z)/len(SYMS),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean())
    for label,a,b in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-28','2026','2028')]:
        q=z.loc[a:b].ic.dropna(); print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan,(q>0).mean())
    for h in [1,5,10,20]:
        ff=px.shift(-h)/px-1; rr=[]
        for d in sig.index:
            a=pd.concat([sig.loc[d],ff.loc[d]],axis=1).dropna()
            if len(a)>=8: rr.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
        rr=pd.Series(rr).dropna(); print('h',h,'n',len(rr),'ic',rr.mean(),'icir',rr.mean()/rr.std())
    # rank turnover on common dates
    ranks=sig.rank(axis=1,pct=True); common=ranks.dropna(how='all').diff().abs().mean(axis=1).dropna()
    print('turnover',common.mean())
if __name__=='__main__': main()
