import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(sym,n=1800):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: f=fn(sym,n)
        except Exception: f=None
        if f is not None and len(f):
            x=f.copy(); x['date']=pd.to_datetime(x['date']); return x.set_index('date').sort_index()['close'].astype(float)
    return None
P={s:get(s) for s in U}; P={s:x for s,x in P.items() if x is not None}
px=pd.concat(P,axis=1).sort_index().ffill(); ret=px.pct_change(); r20=px.pct_change(20); r60=px.pct_change(60); vol=ret.rolling(20).std()*np.sqrt(252)
factors={'agreement_sharpe':np.sign(r20)*np.sign(r60)*(0.6*r20+0.4*r60)/vol,'smooth_trend':(0.6*r20+0.4*r60)/vol,'agreement_raw':np.sign(r20)*np.sign(r60)*(0.6*r20+0.4*r60)}
fr=ret.shift(-1)
for name,F in factors.items():
 obs=[]; turnovers=[]; ns=[]; last=None
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: obs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
  q=pd.concat([F.loc[d],F.loc[:d].iloc[:-1].tail(1).iloc[0]],axis=1).dropna() if len(F.loc[:d].iloc[:-1]) else pd.DataFrame()
  if len(q)>=8: turnovers.append(1-q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 x=pd.Series(dict(obs)); ic=x.mean(); ir=ic/x.std(ddof=1); print(name,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((x>0).mean(),4),'turn',round(np.nanmean(turnovers),4),'coverage',round(len(x)/(len(px)-1),4))
 for label,lo,hi in [('pre',x.index.min(),pd.Timestamp('2026-12-31')),('2027_28',pd.Timestamp('2027-01-01'),pd.Timestamp('2028-12-31')),('2029',pd.Timestamp('2029-01-01'),x.index.max())]:
  y=x[(x.index>=lo)&(x.index<=hi)]
  if len(y)>5: print(' ',label,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6))
print('assets',len(P),'dates',len(px),'through',px.index.max())
