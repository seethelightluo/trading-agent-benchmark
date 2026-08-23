import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is None or len(d)<80: d=get_index_daily_data(s, days=5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
        frames[s]=x
p=pd.DataFrame(frames).sort_index().ffill()
r=p.pct_change()
# Candidate: lagged 20d risk-adjusted trend, with cross-sectional defensive normalization.
# signal at date t only uses closes through t; IC pairs against t+1 close return.
trend=p.pct_change(20)
vol=r.rolling(20).std()
f=trend/vol.replace(0,np.nan)
# damp extremes and favor broad, lower-risk trends (interpretable)
f=f.clip(-5,5)
rows=[]
for i in range(len(p)-1):
    dt=p.index[i]
    z=f.iloc[i]; y=r.iloc[i+1]
    ok=z.notna()&y.notna()
    if ok.sum()>=8:
        rows.append((dt, z[ok].corr(y[ok]), ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'rows',int(a.n.sum()),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.sum()/(len(a)*15),4))
print('daily_ic %.8f icir %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1), (a.ic>0).mean()))
for name,sel in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026',(a.index>='2026-01-01')&(a.index<'2027-01-01')),('2027',(a.index>='2027-01-01')&(a.index<'2028-01-01')),('recent90',a.index>=a.index.max()-pd.Timedelta(days=140)),('recent180',a.index>=a.index.max()-pd.Timedelta(days=260))]:
    q=a[sel]; print(name,len(q), '%.8f %.8f %.3f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()) if len(q)>2 else '')
# decay non-overlapping-ish per-day forward horizons (same signal, horizon return)
for h in [1,3,5,10]:
    vals=[]
    for i in range(len(p)-h):
      z=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1; ok=z.notna()&y.notna()
      if ok.sum()>=8: vals.append(z[ok].corr(y[ok]))
    q=pd.Series(vals).dropna(); print('horizon',h,'dates',len(q),'IC %.8f ICIR %.8f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('last',a.tail(3).to_string())
# save artifact with complete signal
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_2_20280113_defensive_trend_signal.csv',index=False)
