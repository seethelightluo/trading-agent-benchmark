"""Miner 1 research: residual volatility-compression persistence signal."""
import os, glob, json, pickle
import numpy as np
import pandas as pd

END=pd.Timestamp('2035-12-05'); ROOT='../persistent/stock_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# aligned completed daily closes; do not use observations later than runtime cutoff
series={}
for a in assets:
    x=pd.read_csv(f'{ROOT}/{a}.csv',parse_dates=['date']).set_index('date')['close'].replace([np.inf,-np.inf],np.nan)
    series[a]=x.loc[:END]
px=pd.DataFrame(series).sort_index().ffill(); ret=px.pct_change()
# compression: lower recent vol relative to medium vol; residualize against trend and short return
v20=ret.rolling(20,min_periods=15).std(); v60=ret.rolling(60,min_periods=45).std()
raw=-(np.log(v20)-np.log(v60))
m20=px.pct_change(20); m60=px.pct_change(60)
def resid(row):
    y=row.iloc[:,0]; X=row.iloc[:,1:]
    ok=y.notna() & X.notna().all(axis=1)
    out=pd.Series(np.nan,index=row.index)
    if ok.sum()>=8:
        z=X.loc[ok].to_numpy(); z=np.c_[np.ones(len(z)),z]
        out.loc[ok]=y.loc[ok]-z@np.linalg.lstsq(z,y.loc[ok],rcond=None)[0]
    return out
sig=pd.concat([raw.stack(),m20.stack(),m60.stack()],axis=1).groupby(level=0).apply(lambda q: resid(q.droplevel(0))).unstack()
sig.index=pd.to_datetime(sig.index); sig=sig.reindex(columns=assets)
# robust daily cross-sectional Spearman IC
metrics={}
for h in [1,5,10,20]:
    fwd=px.pct_change(h).shift(-h); vals=[]
    for d in sig.index:
        q=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
        if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
    z=np.array(vals); metrics[h]={'dates':len(z),'ic':float(np.mean(z)),'icir':float(np.mean(z)/np.std(z,ddof=1)),'hit':float(np.mean(z>0))}
# turnover, coverage and regimes at decision-relevant h=10
rank=sig.rank(axis=1,pct=True); turnover=float(rank.diff().abs().stack().mean()); coverage=float(sig.notna().mean().mean())
reg={}
fwd=px.pct_change(10).shift(-10)
for n,(lo,hi) in {'2020-2024':('2020-01-01','2024-12-31'),'2025-2029':('2025-01-01','2029-12-31'),'2030-2034':('2030-01-01','2034-12-31'),'2035YTD':('2035-01-01','2035-12-05')}.items():
 a=[]
 for d in sig.loc[lo:hi].index:
  q=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(q)>=8:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 a=np.array(a);reg[n]={'dates':len(a),'ic':float(a.mean()) if len(a) else None,'icir':float(a.mean()/a.std(ddof=1)) if len(a)>1 else None}
# exact signal-library check against recoverable signal panels; panels with same date/asset layout accepted
cors=[]
for p in glob.glob('scripts/*signal.pkl'):
 try:
  z=pd.read_pickle(p)
  if isinstance(z,pd.Series): z=z.unstack()
  if not isinstance(z,pd.DataFrame): continue
  z.index=pd.to_datetime(z.index); common_i=sig.index.intersection(z.index); common_c=sig.columns.intersection(z.columns)
  if len(common_i)*len(common_c)<100: continue
  x=sig.loc[common_i,common_c].stack(); y=z.loc[common_i,common_c].stack(); q=pd.concat([x,y],axis=1).dropna()
  if len(q)>=100: cors.append((abs(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')),p,len(q)))
 except Exception: pass
cors.sort(reverse=True)
out={'cutoff':str(END.date()),'cells':int(sig.notna().sum().sum()),'dates':int(sig.notna().any(axis=1).sum()),'avg_instruments':float(sig.notna().sum(axis=1).mean()),'coverage':coverage,'turnover':turnover,'metrics':metrics,'regimes':reg,'max_library_corr':cors[0] if cors else None}
print(json.dumps(out,indent=2))
sig.to_pickle('scripts/miner_1_20351206_residual_volatility_compression_persistence_20_60_signal.pkl')
with open('scripts/miner_1_20351206_residual_volatility_compression_persistence_20_60_results.json','w') as f:json.dump(out,f,indent=2)
