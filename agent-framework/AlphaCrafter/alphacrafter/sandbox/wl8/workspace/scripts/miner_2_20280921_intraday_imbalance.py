import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-09-19')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); D[s]=x[x.index<=END]
# lagged intraday close/open return reversal: today's completed session gap/intraday imbalance predicts next close return
op=pd.DataFrame({s:D[s].open for s in U}); cl=pd.DataFrame({s:D[s].close for s in U});
imb=(cl/op-1).shift(1); vol=cl.pct_change().rolling(20,min_periods=15).std().shift(1)
sig=-imb/vol
for h in [1,3,5]:
 f=cl.shift(-h)/cl-1; rows=[]
 for d in cl.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append(spearmanr(g.s,g.f).statistic)
 z=pd.Series(rows); q=z.iloc[-180:]
 print(h,len(z), 'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'recent',q.mean(),q.mean()/q.std())
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20280921_intraday_imbalance_signal.csv',index=False)
