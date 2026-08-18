import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-06-12')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:cut]; r=px.pct_change()
# Relative residual reversal: remove contemporaneous cross-asset market move from each asset's
# 20-session return, then reverse the residual and divide by idiosyncratic 20d volatility.
# All inputs are lagged one completed session.
R20=px.pct_change(20); mkt=R20.median(axis=1); resid=R20.sub(mkt,axis=0)
idvol=r.sub(r.median(axis=1),axis=0).rolling(20,min_periods=15).std()
sig=(-resid/(idvol+1e-12)).shift(1)
def run(h,start=None):
 y=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in sig.index:
  if start and d<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals); return len(x),np.mean(ns),np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12)*np.sqrt(len(x)),np.mean(x>0)
for h in [1,5,10,20]: print(h,'full',run(h),'recent',run(h,'2029-06-13'))
rank=sig.rank(axis=1,pct=True); print('coverage',sig.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).mean(),'assets',px.shape[1],'cut',cut.date())
