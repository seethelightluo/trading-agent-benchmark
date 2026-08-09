import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END='2030-01-09'
cand={};base={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END];c=d.close.astype(float);r=c.pct_change(fill_method=None)
 tail=(np.sqrt((r.clip(upper=0)**2).rolling(20,min_periods=15).mean())/r.rolling(20,min_periods=15).std().replace(0,np.nan)).clip(0,2)
 cand[a]=-c.pct_change(5,fill_method=None)*tail
 base[a]=-c.pct_change(5,fill_method=None)/r.rolling(5,min_periods=4).std().replace(0,np.nan)
x=pd.DataFrame(cand).stack();y=pd.DataFrame(base).stack();q=pd.concat([x,y],axis=1).dropna();print('pooled Spearman vs admitted miner_1_volnorm_reversal_5obs:',round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6),'common_cells',len(q))
