import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:'2035-10-14']; r=p.pct_change(); v=r.rolling(20,min_periods=15).std(); disp=r.rolling(20,min_periods=15).std().mean(axis=1)
# residual short reversal, activated only when cross-asset dispersion is above its trailing median
raw=-r.rolling(5,min_periods=5).sum()/v; gate=(disp>disp.rolling(120,min_periods=60).median()).astype(float); f=raw.mul(gate,axis=0)
ics={h:[] for h in [1,5,10,20]}; dates=[]; cov=[]; turns=[]
for i in range(len(p)-21):
 x=f.iloc[i]; n=x.notna().sum()
 if x.nunique(dropna=True)<3: continue
 if n<8: continue
 dates.append(p.index[i]); cov.append(n/15)
 if i: turns.append(np.mean(abs(x.rank(pct=True)-f.iloc[i-1].rank(pct=True))))
 for h in ics:
  z=pd.concat([x,p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: ics[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('dates',len(dates),'avg_n',np.mean(cov)*15,'coverage',np.mean(cov),'turnover',np.nanmean(turns))
for h,a in ics.items():
 a=np.array(a);print('H',h,'obs',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('recent H10',np.nanmean(ics[10][-500:]),np.nanmean(ics[10][-500:])/np.nanstd(ics[10][-500:],ddof=1))
f.to_csv('scripts/miner_1_20351015_dispersion_gate_reversal_signal.csv',index_label='date')
