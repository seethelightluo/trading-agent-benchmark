import pandas as pd, numpy as np, json, os
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2032-12-22')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 xs[s]=d['close'].astype(float)
px=pd.DataFrame(xs).loc[:end]
ret=px.pct_change()
# residual relative to equal-weight benchmark; factor = medium horizon residual trend / downside volatility
bench=ret.mean(axis=1)
res=ret.sub(bench,axis=0)
trend=res.rolling(20,min_periods=18).sum()
down=res.clip(upper=0).pow(2).rolling(40,min_periods=30).mean().pow(.5)
raw=trend/(down+1e-8)
# activation: stressed benchmark or elevated cross-sectional dispersion, lagged by one day
vol=ret.std(axis=1).rolling(20,min_periods=15).mean()
med=vol.rolling(120,min_periods=80).median()
act=(bench.rolling(20,min_periods=20).sum()<0)|(vol>med)
f=raw.where(act,0).shift(1)
fwd=px.shift(-10).div(px)-1
rows=[]
for dt in f.index:
 a=f.loc[dt]; y=fwd.loc[dt]; ok=a.notna()&y.notna()&np.isfinite(a)&np.isfinite(y)
 if ok.sum()>=8:
  x=a[ok].rank(); z=y[ok]
  ic=x.corr(z,method='spearman')
  if np.isfinite(ic): rows.append((dt,ic,ok.sum()))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# avoid last dates lacking full forward horizon
r=r.loc[r.index<=end-pd.Timedelta(days=15)]
mean=r.ic.mean(); sd=r.ic.std(ddof=1); icir=mean/sd if sd else np.nan
print('dates',len(r),'avg_n',r.n.mean(),'coverage',f.notna().sum(axis=1).mean()/15,'IC',mean,'ICIR',icir,'hit',(r.ic>0).mean(),'turnover',f.rank(axis=1).diff().abs().mean().mean()/14)
for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-12-10')]:
 q=r.loc[a:b].ic; print(a[:4]+'-'+b[:4],len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out='scripts/miner_2_20321223_stress_downside_recovery_signal.csv'; pd.DataFrame(f).to_csv(out)
print('artifact',out)
