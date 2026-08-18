import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
    f='../persistent/stock_data/'+a+'.csv'
    if os.path.exists(f):
        d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
        px[a]=d
prices=pd.DataFrame(px).sort_index()
# factor at t uses close through t; forward return t to t+h, validation is strictly observable at t
ret10=prices/prices.shift(10)-1
vol=prices.pct_change().rolling(40,min_periods=30).std()
factor=ret10/vol
# clip each date cross-section
factor=factor.clip(lower=factor.quantile(.05,axis=1),upper=factor.quantile(.95,axis=1),axis=0)
rows=[]
for h in [5,10,20,40]:
    fr=prices.shift(-h)/prices-1
    ics=[]; dates=[]; counts=[]; turns=[]
    for dt in prices.index:
        x=factor.loc[dt]; y=fr.loc[dt]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ics.append(x[m].corr(y[m],method='spearman')); dates.append(dt); counts.append(m.sum())
            if dt in factor.index and dt!=factor.index[0]:
                prev=factor.loc[:dt].iloc[-2]; mm=x.notna()&prev.notna()
                if mm.sum()>=8: turns.append((x[mm].rank().sub(prev[mm].rank()).abs()/max(1,mm.sum())).mean())
    z=pd.Series(ics,index=pd.to_datetime(dates)).dropna(); mean=z.mean(); sd=z.std(ddof=1)
    print(f'h={h} dates={len(z)} avg_n={np.mean(counts):.2f} IC={mean:.8f} ICIR={mean/sd*np.sqrt(252) if sd else np.nan:.8f} hit={(z>0).mean():.4f} coverage={np.mean(counts)/len(assets):.4f} turnover={np.mean(turns):.6f}')
    for label,lo,hi in [('2020-2025','2020-01-01','2025-12-31'),('2026-2030','2026-01-01','2030-12-31'),('2031-2035','2031-01-01','2035-12-31')]:
        q=z.loc[lo:hi]; print(' ',label,'n',len(q),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4) if len(q)>2 else None)
print('assets',len(px),'dates',prices.index.min(),prices.index.max())
# signal artifact for preferred horizon
out=factor.copy(); out.index.name='date'; out.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('../persistent/miner_2_20350302_volscaled_mom10_signal.csv',index=False)
