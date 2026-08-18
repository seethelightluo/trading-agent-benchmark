import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-07-10')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:cut]
r=px.pct_change(); v5=r.rolling(5,min_periods=4).std(); v20=r.rolling(20,min_periods=15).std(); move=px.pct_change(5)
# contrarian move, amplified when recent volatility is compressed versus baseline
sig=(-move*(v5/(v20+1e-12))).shift(1)
def run(h,start=None):
 y=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in sig.index:
  if start and d<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.array(vals); return len(x),np.mean(ns),np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12)*np.sqrt(len(x)),np.mean(x>0)
for h in [1,5,10,20]: print(h,'full',run(h),'recent260',run(h,'2029-07-11'))
rank=sig.rank(axis=1,pct=True)
print('coverage',sig.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).mean(),'assets',px.shape[1],'cut',cut.date())
