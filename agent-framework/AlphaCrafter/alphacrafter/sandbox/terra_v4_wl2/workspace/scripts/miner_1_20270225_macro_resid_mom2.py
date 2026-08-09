import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-02-24')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
M={s:pd.read_csv('../persistent/index_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in ['DXY','VIX']}
px=pd.DataFrame(P).loc[:cut]; mx=pd.DataFrame(M).reindex(px.index).ffill(); ar=px.pct_change(); mr=mx.pct_change()
# lagged 60d macro betas; residualized 20d return, all inputs completed at signal date
bD=ar.rolling(60,min_periods=40).cov(mr['DXY']).div(mr['DXY'].rolling(60,min_periods=40).var(),axis=0)
bV=ar.rolling(60,min_periods=40).cov(mr['VIX']).div(mr['VIX'].rolling(60,min_periods=40).var(),axis=0)
r20=px.pct_change(20); m20=mx.pct_change(20)
f=r20-bD.shift(1).mul(m20['DXY'],axis=0)-bV.shift(1).mul(m20['VIX'],axis=0)
targets={h:px.pct_change(h).shift(-h) for h in [1,5,10]}
for h,t in targets.items():
  vals=[]; ns=[]
  for d in f.index:
    z=pd.concat([f.loc[d],t.loc[d]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.8f ICIR %.8f hit %.4f coverage %.4f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),np.mean(ns)/15))
# regime daily
for name,lo,hi in [('2020-22','2020','2023'),('2023-24','2023','2025'),('2025+','2025','2028')]:
 t=targets[1]; aa=[]
 for d in f.loc[lo:hi].index:
  z=pd.concat([f.loc[d],t.loc[d]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(name,len(aa),round(np.mean(aa),6),round(np.mean(aa)/np.std(aa,ddof=1),6))
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('../persistent/factor_signals_miner_1_20270225_macro_resid_mom2.csv')
print('artifact written')
