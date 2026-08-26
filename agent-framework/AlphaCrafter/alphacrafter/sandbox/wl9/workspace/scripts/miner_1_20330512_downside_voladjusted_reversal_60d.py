import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,5000)
    if x is None or len(x)<150: x=get_index_daily_data(s,5000)
    if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# downside realized volatility, lagged one completed day
rv=((r.clip(upper=0)**2).rolling(60,min_periods=45).mean()**.5)*np.sqrt(252)
trend=p.pct_change(60)
f=(-(trend/(rv+0.05))).shift(1)
rows=[]
for i in range(len(p)-60):
    dt=p.index[i]
    if dt < pd.Timestamp('2020-01-01') or dt>pd.Timestamp('2033-05-11'): continue
    z=f.iloc[i]; fr=p.iloc[i+60]/p.iloc[i]-1
    q=pd.concat([z,fr],axis=1).dropna()
    if len(q)>=8:
      ic=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
      rows.append((dt,ic,len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=a.ic.mean(); sd=a.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('instruments',len(D),'dates',len(a),'avg_n',a.n.mean())
print('IC',mean,'ICIR_daily_scaled',mean/sd if sd else np.nan,'ICIR_ann',icir,'hit', (a.ic>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [10,20,40,60]:
 rr=[]
 for i in range(len(p)-h):
  if p.index[i]<pd.Timestamp('2020-01-01') or p.index[i]>pd.Timestamp('2033-05-11'):continue
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1)],axis=1).dropna()
  if len(q)>=8: rr.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('horizon',h,'dates',len(rr),'IC',np.nanmean(rr),'ICIRdaily',np.nanmean(rr)/np.nanstd(rr,ddof=1))
# artifact
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_1_20330512_downside_voladjusted_reversal_60d_signal.csv',index=False)
