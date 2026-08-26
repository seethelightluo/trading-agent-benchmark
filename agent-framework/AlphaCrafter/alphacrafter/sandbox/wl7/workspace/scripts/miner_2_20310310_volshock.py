import pandas as pd,numpy as np
from scipy.stats import spearmanr
END='2031-03-10'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:END] for s in syms}).sort_index(); R=P.pct_change()
# volatility expansion: reversal of recent return when short volatility sharply exceeds medium volatility
v5=R.rolling(5).std(); v20=R.rolling(20).std(); ret3=P.pct_change(3); sig=-ret3*(v5/v20-1)
fwd=P.pct_change().shift(-1); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for lab,q in [('full',a),('756',a.tail(756)),('252',a.tail(252))]: print(lab,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean())
print('coverage valid dates',len(a),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig.to_csv('scripts/miner_2_20310310_volshock_signal.csv');a.to_csv('scripts/miner_2_20310310_volshock_ic.csv')
