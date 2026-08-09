import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float) for s in U}
P=pd.DataFrame(px).sort_index(); P=P[P.index<=cut]; R5=P.pct_change(5)
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float); v=v[v.index<=cut]
v5=v.pct_change(5).reindex(P.index).ffill().clip(-.5,.5)
f=(-R5.mul(1+v5,axis=0)).where(v5>0, -R5*.5)
Y=P.shift(-1).div(P)-1
obs=[]
for dt in P.index:
 q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(q)>=8: obs.append((dt,q.f.corr(q.y),len(q)))
a=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date'); ic=a.ic
print('cut',cut.date(),'dates',len(ic),'avgN',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for yr,g in ic.groupby(ic.index.year): print('year',yr,'dates',len(g),'IC',g.mean(),'ICIR',g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'rankturn',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('highVIX_dates',(v5>0).sum(),'total_macro',v5.notna().sum())
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_1_20261217_macro_conditioned_reversal_signal.csv',index=False)
