import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-09-05'];r=P.pct_change();f=(-r.rolling(4,min_periods=4).sum()).shift(1);rows=[]
for d in P.index:
 z=pd.concat([f.loc[d],(P.shift(-4)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8:rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
x=pd.Series(rows);print('horizon4 obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4));print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6));f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290906_short_reversal_4d_signal.csv',index=False)
