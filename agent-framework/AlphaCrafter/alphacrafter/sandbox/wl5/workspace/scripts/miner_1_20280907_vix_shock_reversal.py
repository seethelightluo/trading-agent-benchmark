import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000); x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date).dt.normalize(); v=v.set_index('date').close.astype(float).reindex(p.index).ffill()
# Contrarian response after a lagged volatility shock: negative 5d VIX change times negative 20d asset return.
shock=(v.pct_change(5)/v.rolling(60).std()).shift(1).clip(-3,3)
f=-(p.pct_change(20)).mul(shock,axis=0)/(r.rolling(20).std()*np.sqrt(252))
f=f.replace([np.inf,-np.inf],np.nan);fr=f.shift(1); fw=p.shift(-10)/p-1
ics=[]; dates=[]; counts=[]
for d in f.index:
 a,b=fr.loc[d],fw.loc[d];ok=a.notna()&b.notna()
 if ok.sum()>=8:ics.append(a[ok].corr(b[ok],method='spearman'));dates.append(d);counts.append(ok.sum())
ic=pd.Series(ics,index=dates).dropna();turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('candidate=vix_shock_scaled_reversal_20d','dates',len(ic),'avg_n',np.mean(counts),'period',ic.index.min().date(),ic.index.max().date())
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f coverage %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean(),turn,p.notna().mean().mean()))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028')]:
 q=ic.loc[a:b];print(a,len(q),q.mean(),q.mean()/q.std())
print('recent',ic.tail(252).mean(),ic.tail(252).mean()/ic.tail(252).std())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20280907_vix_shock_reversal_signal.csv',index=False)
