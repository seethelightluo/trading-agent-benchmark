import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:'2027-11-03'] for s in S}
p=pd.DataFrame(P).sort_index();r=p.pct_change();f0=(-r.rolling(5,min_periods=5).sum()/r.rolling(20,min_periods=15).std()).clip(-8,8)
for m in ['none','vix_amp','vix_damp']:
 v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill(); stress=(v/v.rolling(60,min_periods=40).median()).clip(.5,2)
 g={'none':pd.Series(1,index=p.index),'vix_amp':stress.pow(.5),'vix_damp':stress.pow(-.5)}[m]
 f=f0.mul(g,axis=0).shift(1).ewm(span=3,min_periods=3,adjust=False).mean();fw=p.shift(-1)/p-1;rows=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(rows);print(m,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
 if m=='vix_amp':f.to_csv('scripts/miner_1_20271104_macro_reversal_signal.csv')
