import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-06-12')
frames={}
for a in assets:
    p='../persistent/stock_data/'+a+'.csv'
    d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
    d=d.loc[:cut]
    # robust close location, with zero-range neutral
    rng=(d.high-d.low).replace(0,np.nan)
    clv=((d.close-d.low)/rng-0.5).fillna(0)
    # factor: persistent close-location pressure, lagged one day
    d['fac']=clv.rolling(5,min_periods=4).mean().shift(1)
    d['fwd10']=d.close.shift(-10)/d.close-1
    frames[a]=d[['fac','fwd10']]
idx=sorted(set().union(*[set(x.index) for x in frames.values()]))
ics=[]; ns=[]; vals=[]
for dt in idx:
    z=pd.DataFrame({a:frames[a].loc[dt] if dt in frames[a].index else [np.nan,np.nan] for a in assets},index=['fac','fwd10']).T.dropna()
    if len(z)>=8:
        ic=z.fac.corr(z.fwd10,method='spearman')
        if pd.notna(ic): ics.append(ic);ns.append(len(z)); vals.append((dt,ic,len(z)))
ics=np.array(ics)
print('cutoff',cut.date(),'dates',len(ics),'assets',len(assets),'avgN',np.mean(ns),'coverage',np.mean(ns)/15)
print('H10 IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),np.mean(ics>0)))
for j,part in enumerate(np.array_split(ics,3)): print('third',j+1,'n',len(part),'ic',part.mean())
for h in [1,5,10,20]:
    arr=[]; nn=[]
    for a in assets:
      d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]
      rng=(d.high-d.low).replace(0,np.nan); clv=((d.close-d.low)/rng-.5).fillna(0)
      fac=clv.rolling(5,min_periods=4).mean().shift(1); fw=d.close.shift(-h)/d.close-1
      frames[a]=pd.DataFrame({'fac':fac,'fw':fw})
    for dt in idx:
      z=pd.DataFrame({a:frames[a].loc[dt] if dt in frames[a].index else [np.nan,np.nan] for a in assets},index=['fac','fw']).T.dropna()
      if len(z)>=8:
       q=z.fac.corr(z.fw,method='spearman')
       if pd.notna(q): arr.append(q)
    arr=np.array(arr); print('H',h,'n',len(arr),'IC %.6f ICIR %.6f'%(arr.mean(),arr.mean()/arr.std(ddof=1)))
# artifact signal latest all dates
out=[]
for a,d in frames.items():
 for dt,row in d.iterrows(): out.append({'date':dt.date(),'asset':a,'signal':row.fac})
pd.DataFrame(out).to_csv('scripts/miner_2_20330613_clv_pressure_signal.csv',index=False)
