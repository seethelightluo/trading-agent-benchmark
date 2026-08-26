import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: recovery-efficiency trend. 20d return weighted by fraction of up days,
# divided by lagged 40d volatility; all inputs end at t-1.
D={}
for s in U:
    x=get_stock_daily_data(s,days=5000)
    if x is not None and len(x)>100:
        z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); z=z.set_index('date').close.astype(float)
        D[s]=z
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# use 20d return and directional persistence, with lag to ensure no current-day data
ret20=p.pct_change(20); up20=(r>0).rolling(20).mean(); vol40=r.rolling(40).std()
f=((ret20*(0.5+up20))/vol40).shift(1)
rows=[]
for h in [1,5,10,20]:
    fr=p.pct_change(h).shift(-h)
    vals=[]
    for dt in f.index:
        a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1]))
    q=pd.Series(vals).dropna(); rows.append((h,len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1), (q>0).mean()))
print('cutoff',p.index.max().date(),'dates',len(p),'avgN',f.notna().sum(axis=1).mean())
for z in rows: print('H%d dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'% (z[0],z[1],z[2],z[4],z[5]))
# turnover rank changes
rank=f.rank(axis=1,pct=True); ch=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f'%(f.notna().mean().mean(),ch.mean()))
# save signal artifact
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20321101_recovery_efficiency_signal.csv')
