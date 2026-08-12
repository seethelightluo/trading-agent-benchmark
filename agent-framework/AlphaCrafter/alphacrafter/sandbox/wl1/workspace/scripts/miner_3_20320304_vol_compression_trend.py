import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=get_stock_daily_data(s, days=1800)
    if d is None or len(d)<120: d=get_index_daily_data(s, days=1800)
    if d is None: continue
    x=d[['date','close']].copy(); x['symbol']=s; rows.append(x)
p=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=np.log(p).diff()
# Candidate: volatility-compression confirmed medium trend. Lag all inputs one day.
ret20=p.pct_change(20); vol20=r.rolling(20).std(); vol60=r.rolling(60).std()
# positive trend with recent volatility below long vol, favor stable persistent trends
f=(ret20/vol20)*(vol20/vol60).replace([np.inf,-np.inf],np.nan)
f=f.shift(1)
fwd=p.shift(-10)/p-1
ics=[]; turnovers=[]; counts=[]
prev=None
for i in range(len(p)-10):
    z=f.iloc[i]; y=fwd.iloc[i]; ok=z.notna()&y.notna()
    if ok.sum()>=8:
        ics.append(z[ok].corr(y[ok])); counts.append(ok.sum())
        rank=z.rank(pct=True); turnovers.append(np.nan if prev is None else (rank-prev).abs().mean())
        prev=rank
ic=np.array([x for x in ics if np.isfinite(x)])
print('candidate=vol_compression_confirmed_trend_20d; dates=%d avg_n=%.3f coverage=%.4f'%(len(ic),np.mean(counts),np.mean(counts)/15))
print('IC=%.6f ICIR=%.6f hit=%.4f turnover=%.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),np.mean(ic>0),np.nanmean(turnovers)))
for h in [1,5,10,20]:
    yy=p.shift(-h)/p-1; aa=[]
    for i in range(len(p)-h):
      ok=f.iloc[i].notna()&yy.iloc[i].notna()
      if ok.sum()>=8: aa.append(f.iloc[i][ok].corr(yy.iloc[i][ok]))
    aa=np.array([v for v in aa if np.isfinite(v)])
    print('decay',h,'%.6f'%aa.mean(),'n',len(aa))
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-03-04')]:
  q=pd.Series(ic,index=p.index[-len(ic):]) if len(ic)==len(p.index) else None
  # recompute by date slice directly
  vals=[]
  for dt in p.loc[a:b].index:
    if dt not in f.index: continue
    ok=f.loc[dt].notna()&fwd.loc[dt].notna()
    if ok.sum()>=8: vals.append(f.loc[dt][ok].corr(fwd.loc[dt][ok]))
  vals=np.array([v for v in vals if np.isfinite(v)])
  print('regime',a[:4],len(vals), 'IC %.6f'%vals.mean() if len(vals) else 'NA')
# save signal artifact wide, date-indexed
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_3_20320304_vol_compression_trend_signal.csv',index=False)
