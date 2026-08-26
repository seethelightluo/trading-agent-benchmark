import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,days=5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
        frames[s]=x
px=pd.DataFrame(frames).sort_index().ffill()
# candidate: 10d relative reversal to defensive anchor, all information lagged one day
ret=np.log(px/px.shift(10))
defens=ret[['XAU','US10Y','CN10Y']].median(axis=1)
f=-(ret.sub(defens,axis=0)).shift(1)
f=f.dropna(how='all')
fr=np.log(px.shift(-1)/px)
ics=[]; rows=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        if np.isfinite(ic): ics.append(ic); rows.append((dt,ic,len(z)))
a=pd.Series(ics)
# rank turnover based on consecutive common dates
r=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
    z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
    if len(z)>=8: turns.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('candidate=defensive_relative_reversal_10d')
print('dates',len(ics),'avgN',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2]/15 for x in rows]))
print('IC %.8f ICIR %.8f hit %.5f turnover %.5f'% (a.mean(),a.mean()/a.std(ddof=1), (a>0).mean(),np.mean(turns)))
# thirds and regime split
for name,sub in [('first',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:])]:
 print(name,'n',len(sub),'ic',sub.mean(),'icir',sub.mean()/sub.std(ddof=1) if len(sub)>1 else np.nan)
# decay horizons
for h in [5,10,20]:
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],np.log(px.shift(-h).loc[dt]/px.loc[dt])],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q)
 print('horizon',h,'n',len(vals),'IC',np.mean(vals),'ICIR',np.mean(vals)/np.std(vals,ddof=1))
# artifact
out=f.copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_3_20320726_defensive_reversal_10d_signal.csv',index=False)
