import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-06-26')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:cut]; r=px.pct_change()
R20=px.pct_change(20); resid=R20.sub(R20.median(axis=1),axis=0); idvol=r.sub(r.median(axis=1),axis=0).rolling(20,min_periods=15).std(); sig=(-resid/(idvol+1e-12)).shift(1)
def run(h,start=None):
 y=px.pct_change(h).shift(-h); v=[]; n=[]
 for d in sig.index:
  if start and d<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z))
 x=np.array(v); return len(x),float(np.mean(n)),float(np.mean(x)),float(np.mean(x)/(np.std(x,ddof=1)+1e-12)*np.sqrt(len(x))),float(np.mean(x>0))
for h in [1,5,10,20]: print(h,'full',run(h),'recent260',run(h,'2029-06-27'))
rank=sig.rank(axis=1,pct=True); print('coverage',float(sig.notna().mean().mean()),'turnover',float(rank.diff().abs().mean(axis=1).mean()),'assets',px.shape[1],'cut',cut.date())
