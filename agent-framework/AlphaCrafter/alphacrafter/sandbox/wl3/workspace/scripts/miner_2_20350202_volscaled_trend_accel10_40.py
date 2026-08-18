import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: volatility-scaled 10d trend acceleration: recent 10d return/realized vol minus 40d return/vol.
# Values at date t use only closes through t; forward return starts t+1.
series={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is not None and len(d)>100:
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.dropna().drop_duplicates('date').set_index('date').sort_index()
        r=np.log(x.close).diff()
        v10=r.rolling(10,min_periods=8).std()*np.sqrt(252)
        v40=r.rolling(40,min_periods=25).std()*np.sqrt(252)
        z10=x.close.pct_change(10)/v10
        z40=x.close.pct_change(40)/v40
        series[s]=z10-z40
# aligned panel and returns
fac=pd.concat(series,axis=1)
fac.to_csv('scripts/miner_2_20350202_volscaled_trend_accel10_40_signal.csv')
prices={}
for s in series:
 d=get_stock_daily_data(s,days=6000); x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); prices[s]=x.drop_duplicates('date').set_index('date').close
pnl=pd.concat(prices,axis=1).sort_index(); fr=pnl.shift(-10)/pnl-1
obs=[]; turnover=[]; counts=[]
for i,dt in enumerate(fac.index[:-10]):
    a=fac.loc[dt]; b=fr.reindex([dt]).iloc[0]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
      obs.append(a[ok].corr(b[ok])); counts.append(int(ok.sum()))
      if i>0:
       prev=fac.iloc[i-1]; kk=ok&(prev.notna())
       if kk.sum()>=8: turnover.append((a[kk].rank(pct=True)-prev[kk].rank(pct=True)).abs().mean())
q=pd.Series(obs).dropna(); n=len(q); mean=q.mean(); sd=q.std(ddof=1)
print({'dates':n,'avg_names':round(float(np.mean(counts)),3),'coverage':round(float(np.mean(counts)/15),4),'IC10':round(float(mean),6),'ICIR10':round(float(mean/sd*np.sqrt(n)),4),'hit':round(float((q>0).mean()),4),'rank_turnover':round(float(np.mean(turnover)),4)})
for k in [120,252,504]:
 z=q.tail(k); print('recent',k,'n',len(z),'IC',round(float(z.mean()),6),'ICIR',round(float(z.mean()/z.std(ddof=1)*np.sqrt(len(z))),4) if len(z)>2 else None)
# decay on common dates
for h in [1,5,10,20]:
 vals=[]
 for dt in fac.index[:-h]:
  a=fac.loc[dt]; b=(pnl.shift(-h)/pnl-1).reindex([dt]).iloc[0]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(a[ok].corr(b[ok]))
 print('decay',h,'n',len(vals),'IC',round(float(np.nanmean(vals)),6))
print('data_end',str(fac.index.max().date()))
