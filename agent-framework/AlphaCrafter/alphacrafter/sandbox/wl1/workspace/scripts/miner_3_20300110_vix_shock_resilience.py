import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:get_stock_daily_data(a,days=5000) for a in U}
V=get_index_daily_data('VIX',days=5000)
px=pd.DataFrame({a:x.set_index('date')['close'] for a,x in D.items()})
vr=V.set_index('date')['pct_change'].rename('vixr')
r=px.pct_change()
# VIX-shock resilience: average asset returns on recent VIX-up days, blended with unconditional return.
# signal at date t uses only t and earlier; forward return starts t+1.
rows=[]
for t in px.index:
    if t not in vr.index: continue
    hist=r.loc[:t].tail(60).join(vr.loc[:t],how='inner').dropna()
    if len(hist)<40: continue
    shock=hist.vixr >= hist.vixr.quantile(.75)
    if shock.sum()<5: continue
    # resilience = conditional shock-day mean - beta-neutralized baseline; stable preference for positive performance in shocks
    vals={a: (hist.loc[shock,a].mean() - hist[a].mean())/ (hist[a].std()+1e-12) for a in U}
    # forward horizons
    try: fut=px.loc[px.index>t].iloc[:20].iloc[-1]/px.loc[px.index>t].iloc[:20].iloc[0]-1
    except: continue
    if len(px.loc[px.index>t].iloc[:20])<20: continue
    for h in [1,5,10,20]:
      fh=px.loc[px.index>t].iloc[:h]
      if len(fh)<h: continue
      for a in U:
        if pd.notna(vals[a]) and pd.notna(fh[a].iloc[-1]/fh[a].iloc[0]-1): rows.append((t,a,vals[a],h,fh[a].iloc[-1]/fh[a].iloc[0]-1))
out=pd.DataFrame(rows,columns=['date','asset','factor','h','fwd'])
metrics=[]
for h,g in out.groupby('h'):
  cs=g.groupby('date').filter(lambda z:len(z)>=8).groupby('date').apply(lambda z:z.factor.corr(z.fwd)).dropna()
  metrics.append((h,len(cs),g.asset.nunique(),cs.mean(),cs.mean()/cs.std(ddof=1), (cs>0).mean()))
print('rows',len(out),'dates',out.date.nunique(),'assets',out.asset.nunique())
print('metrics h dates assets IC ICIR hit')
for x in metrics: print(x)
# save signal artifact at full panel, h20
out[out.h==20][['date','asset','factor']].to_csv('scripts/miner_3_20300110_vix_shock_resilience_signal.csv',index=False)
# recent regimes
x=out[out.h==20].groupby('date').filter(lambda z:len(z)>=8)
for label,lo,hi in [('2020-25','2020-01-01','2025-12-31'),('2026-28','2026-01-01','2028-12-31'),('2029','2029-01-01','2029-12-31')]:
 zz=x[(x.date>=lo)&(x.date<=hi)]; z=pd.Series({d:g.factor.corr(g.fwd) for d,g in zz.groupby('date')}).dropna(); print(label,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan,(z>0).mean())
print('coverage',out[out.h==20].asset.nunique()/15,'turnover',out[out.h==20].sort_values('date').groupby('date').apply(lambda z:z.set_index('asset').factor.rank(pct=True)).groupby(level=1).diff().abs().mean())
