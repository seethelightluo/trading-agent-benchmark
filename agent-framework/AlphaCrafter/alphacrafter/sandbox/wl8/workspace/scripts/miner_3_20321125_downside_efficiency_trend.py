import numpy as np, pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: downside-adjusted trend persistence. Positive drift is rewarded, downside volatility penalized;
# multiply by path efficiency to distinguish smooth trends from noisy gains.
D={}
for s in U:
    try: x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
    except FileNotFoundError: continue
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
        D[s]=x['close'].astype(float).replace(0,np.nan)
p=pd.DataFrame(D).sort_index()
r=np.log(p).diff()
# lag-safe at date t: all rolling calculations naturally end at t, then forward return starts t+1
mu=r.rolling(20,min_periods=15).mean()
down=np.sqrt((r.clip(upper=0)**2).rolling(20,min_periods=15).mean())
eff=(p.diff(20).abs()/(p.diff().abs().rolling(20,min_periods=15).sum())).clip(0,3)
f=(mu/(down+1e-8)*eff).replace([np.inf,-np.inf],np.nan)
fr=np.log(p.shift(-10)/p)
ics=[]; vals=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        if pd.notna(ic): ics.append((dt,ic,len(z))); vals.append(f.loc[dt])
i=pd.Series([x[1] for x in ics],index=[x[0] for x in ics])
print('dates',len(i),'avgN',np.mean([x[2] for x in ics]),'coverage',np.mean([x[2] for x in ics])/15)
print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/(i.std(ddof=1)+1e-12), (i>0).mean()))
for n,label in [(365,'recent365'),(750,'recent750'),(1260,'recent1260')]:
 q=i.tail(n); print(label,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)))
# turnover as rank changes
ranks=f.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna(); print('turnover',turn.mean())
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p); a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
# save artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20321125_downside_efficiency_trend_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i.values}).to_csv('scripts/miner_3_20321125_downside_efficiency_trend_ic.csv',index=False)
