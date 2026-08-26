import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2033-08-31')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:cutoff]
r=p.pct_change(); vol=r.rolling(30).std()*np.sqrt(252)
# curvature: recent trend relative to slower trend, normalized by lagged risk; all inputs lagged
r20=p.shift(1)/p.shift(21)-1
r60=p.shift(1)/p.shift(61)-1
f=(r20-r60/3)/vol.shift(1)
f=f.sub(f.median(axis=1),axis=0).rolling(5,min_periods=5).mean()
fwd=p.shift(-10)/p-1
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for label,q in [('full',ic),('recent365',ic.tail(365)),('recent730',ic.tail(730)),('pre_recent',ic.iloc[:-365])]:
 m=q.ic.mean(); sd=q.ic.std(ddof=1); print(label,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),4))
rank=f.rank(axis=1,pct=True); print('coverage',round(f.notna().stack().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),6)); print('period',ic.index.min(),ic.index.max())
print('decay')
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; rr=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print(h,round(np.nanmean(rr),6),len(rr))
# save signal only after research (script always emits artifact)
f.to_csv('scripts/miner_2_20330901_trend_curvature_volnorm_signal.csv')
