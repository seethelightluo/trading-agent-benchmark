import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().groupby(level=0).last().loc[:'2034-07-05']
R=np.log(C).diff()
# Lagged short reversal, normalized by idiosyncratic (cross-sectional demeaned) volatility.
csmean=R.mean(axis=1)
res=R.sub(csmean,axis=0)
base=res.rolling(10,min_periods=8).mean().shift(1)
vol=res.rolling(20,min_periods=12).std().shift(1).replace(0,np.nan)
# Breadth transition: dispersion rising versus its lagged 20-day average; bounded gate.
disp=R.std(axis=1).shift(1)
transition=(disp/disp.rolling(20,min_periods=10).mean()).clip(.5,2.0)
raw=(-base/vol).mul(transition,axis=0)
f=raw.rank(axis=1,pct=True); f=f.sub(f.mean(axis=1),axis=0)
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[]; n=[]; dates=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z)); dates.append(d)
 return pd.Series(a,index=dates),pd.Series(n,index=dates)
i,nn=calc(q(10)); print('end',C.index.max().date(),'dates',len(i),'avgN',round(nn.mean(),3),'coverage',round(nn.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]: print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20340706_breadth_transition_residual_reversal10_wide_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20340706_breadth_transition_residual_reversal10_wide_ic.csv')
