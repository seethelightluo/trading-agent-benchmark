import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# downside shock asymmetry: assets with relatively benign downside-tail share should outperform
D={s:get_stock_daily_data(s,days=4000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
ret=np.log(px).diff()
# signal at date t, explicitly lagged one day; lower downside asymmetry ranks higher
# rolling downside semivariance relative to upside, with smoothing and volatility normalization
for window in [20,40,60]:
  down=ret.clip(upper=0).pow(2).rolling(window,min_periods=max(10,window//2)).mean()
  up=ret.clip(lower=0).pow(2).rolling(window,min_periods=max(10,window//2)).mean()
  raw=-(down/(up+1e-8)).replace([np.inf,-np.inf],np.nan)
  sig=raw.shift(1)
  print('\nWINDOW',window)
  for h in [1,3,5,10]:
    fwd=px.pct_change(h).shift(-h)
    vals=[]; ns=[]; dates=[]
    for dt in sig.index:
      x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
      if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
        vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
    a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2) if ns else 0,'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),3),'cov',round(np.mean(ns)/len(U),3))
  # regime/year checks at daily horizon
  fwd=px.pct_change(1).shift(-1); vals=[]
  for dt in sig.index:
    z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(z)>=8: vals.append((dt,pd.Series(z.iloc[:,0]).corr(pd.Series(z.iloc[:,1]),method='spearman')))
  q=pd.DataFrame(vals,columns=['date','ic']); q['year']=q.date.dt.year
  print('YEAR',q.groupby('year').ic.mean().round(4).to_dict())
