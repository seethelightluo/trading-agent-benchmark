import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s, days=4000) for s in U}
close=pd.concat({s:d.set_index('date')['close'] for s,d in raw.items() if d is not None},axis=1).sort_index().ffill()
# interpretable trend-consistency factor, all inputs end t-1
ret=close.pct_change()
r20=close.shift(1).pct_change(20)
cons=ret.shift(1).rolling(20,min_periods=16).mean()/ret.shift(1).rolling(20,min_periods=16).std()
# signal is medium momentum rewarded only when daily direction is consistent
sig=(r20 * (cons.clip(-3,3))).replace([np.inf,-np.inf],np.nan)
# rank-like cross sectional demean
fwd=close.shift(-10)/close-1
rows=[]
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),z.iloc[:,0].corr(z.iloc[:,1],method='pearson')))
df=pd.DataFrame(rows,columns=['date','n','ic_s','ic_p']).set_index('date')
print('dates',len(df),'avgN',df.n.mean(),'coverage',sig.notna().sum(axis=1).mean()/len(U))
for h in [1,5,10,20]:
    yy=close.shift(-h)/close-1; rr=[]
    for dt in sig.index:
      z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
      if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=pd.Series(rr).dropna(); print('H',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'n',len(a))
print('H10 thirds',df.ic_s.groupby(pd.qcut(np.arange(len(df)),3,labels=False)).mean().tolist())
# turnover of cross-sectional ranks
r=sig.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20331212_consistency_momentum_signal.csv',index=False)
print('artifact rows',len(out))
