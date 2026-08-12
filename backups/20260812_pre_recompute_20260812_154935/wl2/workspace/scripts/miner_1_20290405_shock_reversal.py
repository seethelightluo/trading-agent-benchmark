import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def q(s):
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=1800)
 return d
raw={s:q(s) for s in U};raw={s:d for s,d in raw.items() if d is not None};px=pd.DataFrame({s:d.set_index('date').close for s,d in raw.items()})
r=px.pct_change(); v=r.rolling(20,min_periods=10).std(); breadth=r.rolling(20,min_periods=10).mean().gt(0).mean(axis=1)
# one-day contrarian shock, volatility scaled; broad stress suppresses noisy reversal
f=(-r/v).mul((1-0.7*(breadth.sub(.5).abs()*2).clip(0,1)),axis=0).shift(1)
for h in [1,5,10,20]:
 fw=px.pct_change(h).shift(-h);a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 a=np.array(a);print('h%d IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(h,np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),np.mean(a>0),len(a),np.mean(ns)))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum(axis=1).mean()/15)
f.to_csv('scripts/miner_1_20290405_shock_reversal_signal.csv',index_label='date')
