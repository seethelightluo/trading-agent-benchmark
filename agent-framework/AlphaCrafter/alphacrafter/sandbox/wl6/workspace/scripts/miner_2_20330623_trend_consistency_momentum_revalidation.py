import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2033-06-22'); horizons=[5,10,20,40]; xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change()
 fac=c.pct_change(20)*(2*r.gt(0).rolling(20).mean()-1)
 xs[s]=pd.DataFrame({'fac':fac, **{str(h):c.shift(-h)/c-1 for h in horizons}})
dates=sorted(set().union(*[x.index for x in xs.values()]))
for h in horizons:
  ics=[]; ns=[]; turns=[]; prev={}; obsdates=[]
  for dt in dates:
   if dt>end: continue
   a=[]; b=[]
   for s,x in xs.items():
    if dt in x.index and pd.notna(x.loc[dt,'fac']) and pd.notna(x.loc[dt,str(h)]): a.append(x.loc[dt,'fac']); b.append(x.loc[dt,str(h)])
   if len(a)>=8:
    q=spearmanr(a,b).statistic
    if np.isfinite(q): ics.append(q); ns.append(len(a)); obsdates.append(dt)
   cur={s:xs[s].loc[dt,'fac'] for s in U if dt in xs[s].index and pd.notna(xs[s].loc[dt,'fac'])}
   if prev:
    common=set(prev)&set(cur)
    if len(common)>=8:
     ra=pd.Series({s:prev[s] for s in common}).rank(); rb=pd.Series({s:cur[s] for s in common}).rank(); turns.append(np.mean(abs(ra-rb))/(len(common)-1))
   prev=cur
  z=np.array(ics); ic=z.mean(); icir=ic/z.std(ddof=1)
  print(f'h {h} dates {len(z)} avg_n {np.mean(ns):.3f} coverage {np.mean(ns)/15:.4f} IC {ic:.6f} ICIR {icir:.6f} hit {np.mean(z>0):.4f} turnover {np.mean(turns):.4f} start {obsdates[0].date()} end {obsdates[-1].date()}')
  if h==10:
   for yr in range(2020,2034):
    vals=[v for d,v in zip(obsdates,z) if d.year==yr]
    if vals: print('year',yr,'n',len(vals),'ic',f'{np.mean(vals):.6f}')
