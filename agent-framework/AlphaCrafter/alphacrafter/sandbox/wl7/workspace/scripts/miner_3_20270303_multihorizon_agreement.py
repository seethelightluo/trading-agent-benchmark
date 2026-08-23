import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def main():
    px={}
    for s in U:
        d=get_stock_daily_data(s, days=2600)
        if d is not None and len(d)>100:
            x=d.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
            px[s]=x['close'].astype(float)
    close=pd.DataFrame(px).sort_index()
    # candidate: agreement of 10d and 30d momentum, scaled by 20d volatility; lagged signal
    r1=close.pct_change(10); r2=close.pct_change(30); vol=close.pct_change().rolling(20,min_periods=15).std()
    # signed agreement: reward trend only when horizons agree, otherwise modest reversal/zero
    raw=((r1.abs()+r2.abs())/2)/(vol+1e-12) * np.sign(r1)* (np.sign(r1)==np.sign(r2)).astype(float)
    sig=raw.shift(1)
    fwd={h:close.shift(-h)/close-1 for h in [1,5,10,20]}
    rows=[]
    for dt in sig.index:
        for h in [1,5,10,20]:
            a=pd.concat([sig.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
            if len(a)>=8:
                ic=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
                if np.isfinite(ic): rows.append((dt,h,ic,len(a)))
    z=pd.DataFrame(rows,columns=['date','h','ic','n'])
    print('assets',len(close.columns),'dates',len(close),'range',close.index.min(),close.index.max())
    print('nonnull sig',sig.notna().sum().describe().to_dict(),'sig rows',sig.notna().any(axis=1).sum(),'fwd1',fwd[1].notna().sum().describe().to_dict())
    print('candidate multi-horizon agreement momentum (lag 1); observations >=8')
    for h in [1,5,10,20]:
        q=z[z.h==h].copy(); mean=q.ic.mean(); sd=q.ic.std(ddof=1)
        print('h',h,'dates',len(q),'avg_n',q.n.mean(),'IC',mean,'ICIR',mean/sd*np.sqrt(252) if sd else np.nan,'hit', (q.ic>0).mean())
    # rank turnover using valid dates
    ss=sig.dropna(how='all'); ranks=ss.rank(axis=1,pct=True); changes=[]
    for i in range(1,len(ranks)):
        a=ranks.iloc[i-1]; b=ranks.iloc[i]; common=a.notna()&b.notna()
        if common.sum()>=8: changes.append((a[common]-b[common]).abs().mean())
    print('coverage',sig.notna().sum(axis=1).mean()/len(U),'rank_turnover',np.mean(changes),'valid dates',len(changes))
    # regimes for 1d
    q=z[z.h==1]
    for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
        t=q[(q.date>=lo)&(q.date<=hi+'-12-31')]
        print('regime',lo,hi,'dates',len(t),'IC',t.ic.mean())
if __name__=='__main__': main()
