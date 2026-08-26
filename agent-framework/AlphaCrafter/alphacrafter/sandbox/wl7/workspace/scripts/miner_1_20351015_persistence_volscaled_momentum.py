import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index().loc[:'2035-10-14']
r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); ret60=p.shift(1)/p.shift(61)-1
# trend persistence: signed fraction of positive daily returns, applied as confidence to medium trend, vol scaled
pos=(r.rolling(40,min_periods=30).mean()/r.abs().rolling(40,min_periods=30).mean())
f=(ret60/vol)*pos
# cross-sectional demean is immaterial to IC but stabilizes scale
ics={h:[] for h in [1,5,10,20]}; dates=[]; cov=[]; turns=[]
for i in range(len(p)-21):
 d=p.index[i]; x=f.iloc[i]; n=x.notna().sum()
 if n<8: continue
 dates.append(d); cov.append(n/15)
 if len(dates)>1: turns.append(np.mean(np.abs(x.rank(pct=True)-f.iloc[i-1].rank(pct=True))))
 for h in ics:
  y=p.iloc[i+h]/p.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: ics[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('dates',len(dates),'avg_n',np.mean(cov)*15,'coverage',np.mean(cov),'turnover',np.nanmean(turns))
for h,v in ics.items():
 a=np.array(v); print('H',h,'obs',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
for label,lo in [('early','2020'),('mid','2025'),('recent','2030')]:
 a=[]
 for d,x in zip(dates,ics[10]):
  if str(d)[:4]>=lo:a.append(x)
 print(label,'H10',len(a),np.mean(a) if a else np.nan, np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
f.to_csv('scripts/miner_1_20351015_persistence_volscaled_momentum_signal.csv',index_label='date')
