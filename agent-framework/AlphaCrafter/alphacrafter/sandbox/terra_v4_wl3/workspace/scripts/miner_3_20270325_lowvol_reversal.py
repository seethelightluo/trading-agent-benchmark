import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-03-25')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();px[s]=d.loc[d.index<=cut,'close']
P=pd.DataFrame(px).dropna(how='all');R=P.pct_change();
# defensive low-volatility cross-sectional rank, with short-term return residual removed
vol=R.rolling(10).std(); mom=R.rolling(3).sum(); F=-vol.rank(axis=1,pct=True) - .25*mom.rank(axis=1,pct=True)
for h in [1,5,10]:
 fr=P.shift(-h).div(P)-1; vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 print(h,len(vals),np.mean(ns),np.mean(vals),np.mean(vals)/np.std(vals,ddof=1),np.mean(np.array(vals)>0))
print('coverage',F.notna().sum().sum()/(len(F)*15),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
out=F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20270325_lowvol_reversal_signal.csv',index=False)
