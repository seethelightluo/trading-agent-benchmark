import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: downside/upside asymmetry (20d mean signed return divided by downside deviation), lagged.
# Tests whether persistent positive-vs-negative return asymmetry predicts next-day cross-section.
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is None or len(d)<100: d=get_index_daily_data(s, days=2600)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d
close=pd.DataFrame({s:d['close'] for s,d in frames.items()}); op=pd.DataFrame({s:d['open'] for s,d in frames.items()})
ret=close.pct_change()
# asymmetry: upside mean less downside mean, scaled by total vol; all inputs through t
up=ret.clip(lower=0).rolling(20,min_periods=15).mean()
dn=(-ret.clip(upper=0)).rolling(20,min_periods=15).mean()
vol=ret.rolling(20,min_periods=15).std()
f=(up-dn)/(vol+1e-12)
# mild cross-sectional demeaning is irrelevant to rank IC but stabilizes values
f=f.replace([np.inf,-np.inf],np.nan)
fwd=close.pct_change().shift(-1)
ics=[]; dates=[]; nms=[]
for dt in f.index:
    x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); nms.append(len(z))
ics=pd.Series(ics,index=dates).dropna()
def stat(x): return (x.mean(), x.mean()/x.std(ddof=1), (x>0).mean(),len(x))
print('assets',len(frames),'dates',len(ics),'avg_names',np.mean(nms),'daily',stat(ics))
for h in [5,10]:
    yy=close.pct_change(h).shift(-h); a=[]
    for dt in f.index:
      z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
      if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=pd.Series(a).dropna();print('horizon',h,stat(a))
# date turnover based on rank ordering / normalized signal changes
r=f.rank(axis=1,pct=True); turnover=r.diff().abs().mean(axis=1).dropna().mean()
print('coverage',f.notna().mean().mean(),'turnover',turnover)
for name,sl in [('2020-22',slice('2020','2022')),('2023-24',slice('2023','2024')),('2025-26',slice('2025','2026'))]:
 x=ics.loc[sl]; print(name,stat(x))
