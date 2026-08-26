import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().ffill(); r=P.pct_change(); disp=r.rolling(5).std().mean(axis=1); raw=-(P.shift(1)/P.shift(11)-1)/(r.rolling(30).std().shift(1)*np.sqrt(10)); y=P.shift(-20)/P-1
for gate,x in [('high75',raw.where(disp.gt(disp.rolling(120).quantile(.75).shift(1)),np.nan)),('low25',raw.where(disp.lt(disp.rolling(120).quantile(.25).shift(1)),np.nan))]:
 x=x.sub(x.mean(axis=1),axis=0); I=[]; cov=[]; turn=[]; dates=[]; prev=None
 for d in x.index:
  ok=x.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:
   I.append(spearmanr(x.loc[d][ok],y.loc[d][ok]).statistic);cov.append(ok.mean());dates.append(d)
   if prev is not None:
    z=prev.notna()&x.loc[d].notna(); turn.append(np.mean(abs(x.loc[d][z].rank(pct=True)-prev[z].rank(pct=True))))
   prev=x.loc[d]
 I=np.array(I); print(gate,len(I),np.mean(I),np.mean(I)/np.std(I,ddof=1),np.mean(I>0),np.mean(cov),np.mean(turn))
 for yr in ['2020','2024','2027']:
  z=I[[str(d)[:4]==yr for d in dates]]
  if len(z):print(yr,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else 0)
 x.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20281204_'+gate+'_signal.csv',index=False)
