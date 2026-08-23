import numpy as np
import pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
    frames[s]=d.drop_duplicates('date').set_index('date').close.astype(float).sort_index()
p=pd.concat(frames,axis=1).sort_index(); r=p.pct_change()
sig=p.pct_change(10)/r.where(r<0,0).pow(2).rolling(40).mean().pow(.5)
for h in [5,10,20]:
    fwd=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
    for dt in sig.index:
        a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(a)>=8:
            vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(a))
    z=pd.Series(vals,index=dates).dropna(); ic=z.mean(); icir=ic/z.std(ddof=1)*np.sqrt(len(z))
    print(f'h={h} dates={len(z)} avg_n={np.mean(ns):.2f} IC={ic:.8f} ICIR={icir:.6f} hit={(z>0).mean():.4f}')
valid=sig.notna().sum(axis=1)/len(U); rank=sig.rank(axis=1,pct=True)
print(f'coverage={valid.mean():.6f} turnover_proxy={rank.diff().abs().mean(axis=1).dropna().mean():.8f} instruments={len(frames)} span={p.index.min().date()}..{p.index.max().date()}')
