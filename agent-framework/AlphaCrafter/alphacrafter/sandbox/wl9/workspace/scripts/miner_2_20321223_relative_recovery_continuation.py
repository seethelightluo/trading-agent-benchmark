import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for f in (get_stock_daily_data,get_index_daily_data):
        try:
            x=f(symbol=s,days=5000)
            if x is not None and len(x)>300:return x[['date','close']]
        except Exception: pass
    return None
p={s:get(s) for s in U}; p={s:x for s,x in p.items() if x is not None}
c=pd.concat([x.set_index('date').close.rename(s) for s,x in p.items()],axis=1).sort_index().ffill()
r20=c/c.shift(20)-1; r60=c/c.shift(60)-1; vol=c.pct_change().rolling(20).std()*np.sqrt(252)
gate=r60.gt(r60.median(axis=1),axis=0)
sig=(r20/vol).where(gate).shift(1)
for h in [10,20,40,60]:
    f=c.shift(-h)/c-1; qs=[]; ns=[]
    for d in sig.index:
        z=pd.concat([sig.loc[d].rename('s'),f.loc[d].rename('f')],axis=1).dropna()
        if len(z)>=8: qs.append(z.s.corr(method='spearman').iloc[0,1]); ns.append(len(z))
    q=pd.Series(qs).dropna(); print(h,'dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('assets',len(p),'dates',len(c),'range',c.index.min(),c.index.max(),'coverage',sig.notna().mean().mean())
rank=sig.rank(axis=1,pct=True); common=rank.notna().sum(axis=1)>=8
print('turnover_proxy',rank[common].diff().abs().mean(axis=1).dropna().mean())
sig.to_csv('scripts/miner_2_20321223_relative_recovery_continuation_signal.csv',index_label='date'); print('artifact scripts/miner_2_20321223_relative_recovery_continuation_signal.csv')
