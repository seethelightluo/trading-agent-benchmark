import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-05-13')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 d=d.loc[:cut]
 xs[s]=d['close'].astype(float)
prices=pd.DataFrame(xs).sort_index()
# inverse 20d realized volatility, computed through t; forward 10 trading day return
r=np.log(prices/prices.shift(1)); vol=r.rolling(20,min_periods=15).std(); fac=-vol
fwd=prices.shift(-10)/prices-1
ics=[]; nobs=[]; turnover=[]
prev=None
for dt in prices.index:
 a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); nobs.append(len(z))
  ranks=z.iloc[:,0].rank(pct=True)
  if prev is not None: turnover.append(np.mean((ranks-prev).abs()))
  prev=ranks
x=np.array(ics); print({'factor':'inverse_20d_vol','horizon':10,'cutoff':str(cut.date()),'dates':len(x),'avg_instruments':round(float(np.mean(nobs)),2),'coverage':round(float(np.mean(nobs))/15,4),'IC':round(float(np.nanmean(x)),6),'ICIR':round(float(np.nanmean(x)/np.nanstd(x,ddof=1)),6),'hit_ratio':round(float(np.mean(x>0)),4),'turnover_rank':round(float(np.nanmean(turnover)),4),'first':str(prices.index.min().date()),'last':str(prices.index.max().date())})
# regime halves
for label,sel in [('early',x[:len(x)//2]),('late',x[len(x)//2:])]: print(label,round(float(np.nanmean(sel)),6),round(float(np.nanmean(sel)/np.nanstd(sel,ddof=1)),6),len(sel))
