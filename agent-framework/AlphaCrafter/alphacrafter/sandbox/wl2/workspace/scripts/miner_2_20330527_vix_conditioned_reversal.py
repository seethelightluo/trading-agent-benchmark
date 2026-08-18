import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Macro-conditioned: lagged 5d residual reversal, active when lagged VIX is above its 60d median and rising; tests stress-specific reversal
px={s:get_stock_daily_data(s,4000) for s in U}
vix=get_index_daily_data('VIX',4000)
for s in U: px[s]['date']=pd.to_datetime(px[s].date); px[s]=px[s].set_index('date')
vix['date']=pd.to_datetime(vix.date); vix=vix.set_index('date')['close']
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]))
rows=[]
for d in dates:
  vals={s:px[s].loc[d,'close'] for s in U}
  # all signals based through d, then forward return d->d+H
  hist={s:px[s].loc[:d,'close'] for s in U}
  r5=pd.Series({s:(hist[s].iloc[-1]/hist[s].iloc[-6]-1 if len(hist[s])>=25 else np.nan) for s in U})
  vol=pd.Series({s:hist[s].pct_change().iloc[-21:-1].std() for s in U})
  med=r5.median(); sig=-(r5-med)/(vol.replace(0,np.nan))
  vv=vix.loc[:d].dropna()
  active=len(vv)>=61 and vv.iloc[-1]>vv.iloc[-61:].median() and vv.iloc[-1]>vv.iloc[-2]
  if not active: sig[:]=np.nan
  for H in [1,3,5,10]:
    for s in U:
      if s in sig.index and pd.notna(sig[s]):
       fut=px[s].loc[px[s].index>d,'close']
       if len(fut)>=H: rows.append((d,s,float(sig[s]),H,float(fut.iloc[H-1]/vals[s]-1)))
df=pd.DataFrame(rows,columns=['date','s','sig','h','fut'])
df[df.h==1][['date','s','sig']].to_csv('scripts/miner_2_20330527_vix_conditioned_reversal_signal.csv',index=False)
print('dates',len(dates),'rows',len(df),'active_dates',df.date.nunique(),'avg_n',df.groupby('date').size().mean())
for h,g in df.groupby('h'):
  ics=g.groupby('date').apply(lambda x:x.sig.corr(x.fut),include_groups=False).dropna()
  print(h,'IC',g.sig.corr(g.fut),'ICIR',ics.mean()/ics.std(ddof=1),'hit', (ics>0).mean(),'nobs',len(ics),'coverage',len(g)/(len(dates)*15),'turnover approx active')
  for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
   z=ics[(ics.index>=a)&(ics.index<=b+'-12-31')]; print(a,b,len(z),round(z.mean(),5) if len(z) else None,round(z.mean()/z.std(ddof=1),4) if len(z)>1 else None)
