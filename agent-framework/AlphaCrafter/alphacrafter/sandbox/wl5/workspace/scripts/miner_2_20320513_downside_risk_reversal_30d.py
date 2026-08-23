import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: medium-horizon downside-risk-adjusted reversal. Fade 30d return, penalizing only downside volatility.
frames={}
for s in U:
    d=get_stock_daily_data(s, days=3300)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d.close.astype(float)
p=pd.concat(frames,axis=1).sort_index()
r=p.pct_change()
# causal factor at t: negative 30d return / downside deviation over trailing 45 sessions
ret=p/p.shift(30)-1
neg=r.where(r<0,0.0)
down=neg.rolling(45,min_periods=25).std()*np.sqrt(252)
f=-(ret/(down+1e-8))
# winsorize cross section, but retain interpretation
f=f.clip(lower=f.quantile(.05,axis=1), upper=f.quantile(.95,axis=1), axis=0)
# forward 10 trading day returns
fr=p.shift(-10)/p-1
ics=[]; turnovers=[]; counts=[]
for dt in f.index:
    x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
        counts.append(len(z))
# turnover as mean rank displacement / N on consecutive valid dates
ranks=f.rank(axis=1,pct=True); a=ranks.diff().abs().mean(axis=1)
turn=a.dropna().mean()
ics=pd.Series(ics).dropna()
print('candidate=downside_risk_adjusted_reversal_30d')
print('dates',len(ics),'mean_instruments',np.mean(counts),'coverage_pct',np.mean(counts)/15*100)
print('IC %.8f ICIR %.8f hit %.6f turnover %.8f'%(ics.mean(),ics.mean()/ics.std(ddof=1), (ics>0).mean(),turn))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-05-12')]:
    q=ics[(ics.index.astype(str)>=lo)&(ics.index.astype(str)<=hi)]
    # ics index is integer due series construction, so regime omitted
# write signal artifact with date index explicitly and values
out=f.loc[:'2032-05-12'].copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20320513_downside_risk_reversal_30d_signal.csv')
