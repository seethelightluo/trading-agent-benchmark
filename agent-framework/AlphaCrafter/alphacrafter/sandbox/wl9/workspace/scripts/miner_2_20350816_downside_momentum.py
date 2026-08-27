import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# downside-risk-adjusted medium horizon momentum, lagged one completed session
allx={}
for s in U:
    d=get_stock_daily_data(s, days=10000)
    if d is None or len(d)<150: d=get_index_daily_data(s, days=10000)
    if d is not None and len(d)>0:
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').sort_index()
        allx[s]=x.close.astype(float)
p=pd.DataFrame(allx).sort_index()
r=p.pct_change()
# downside deviation: RMS of negative daily returns, annualization cancels cross-sectionally
neg=r.where(r<0,0.0)
down=np.sqrt((neg**2).rolling(60,min_periods=40).mean())
# trend persistence: medium momentum rewarded, penalize downside risk; lag avoids lookahead
sig=(p.pct_change(60)/(down* np.sqrt(60)+0.005)).shift(1)
# robust clipping cross section
sig=sig.clip(-20,20)

rows=[]
for h in [5,10,20,40,60]:
    ics=[]; ns=[]; dates=[]
    fwd=p.shift(-h)/p-1
    for dt in sig.index:
        a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:
            ics.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a)); dates.append(dt)
    z=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
    print(f'H{h} IC {z.mean():+.6f} ICIR {z.mean()/z.std(ddof=1):+.6f} hit {(z>0).mean():.4f} dates {len(z)} avgN {np.mean(ns):.2f}')
    if h==10:
        # turnover of ranks/signals over 10 sessions
        ranks=sig.rank(axis=1,pct=True)
        turn=(ranks-ranks.shift(10)).abs().mean(axis=1).dropna().mean()
        print(f'H10 coverage {sig.notna().mean().mean():.6f} turnover10 {turn:.6f}')
        for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
            q=z[(z.index>=lo+'-01-01')&(z.index<=hi+'-12-31')]
            if len(q): print(f'REG {lo}-{hi} n {len(q)} IC {q.mean():+.6f} ICIR {q.mean()/q.std(ddof=1):+.6f}')
# save reproducible latest artifact
out=sig.reset_index().rename(columns={'date':'date'})
out.to_csv('scripts/miner_2_20350816_downside_momentum_signal.csv',index=False)
print('DATA dates',len(p),'assets',len(p.columns),'range',p.index.min(),p.index.max())
