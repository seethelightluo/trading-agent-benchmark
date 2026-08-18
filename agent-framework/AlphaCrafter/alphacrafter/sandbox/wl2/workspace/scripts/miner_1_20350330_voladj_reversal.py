import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']) for s in U}
# align close and returns
close=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index()
ret=np.log(close).diff()
# 60d reversal risk adjusted; lag via rolling naturally uses t, then signal evaluated t -> fwd after t
def evalf(f,h):
 f=f.replace([np.inf,-np.inf],np.nan); fr=close.pct_change(h).shift(-h)
 vals=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(vals); return len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)
# avoid lookahead: signal at t based through t-1
r60=ret.rolling(60).sum().shift(1)
v60=ret.rolling(60).std().shift(1)*np.sqrt(60)
f=-r60/v60
for h in [5,10,20,40]: print('60d_vol_adj_reversal',h,evalf(f,h))
# conditional: reversal only when 20d return negative, otherwise zero (still rank tie issue)
f2=f.where(r60<0,0)
for h in [10,20,40]: print('conditional_negative',h,evalf(f2,h))
print('dates',len(close),'assets',close.shape[1])
