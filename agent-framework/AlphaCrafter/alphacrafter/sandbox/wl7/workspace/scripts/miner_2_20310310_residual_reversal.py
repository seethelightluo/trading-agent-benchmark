import pandas as pd, numpy as np
from scipy.stats import spearmanr
END='2031-03-10'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:END] for s in syms}
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1); m5=(1+m).rolling(5).apply(np.prod,raw=True)-1
r5=P.pct_change(5); sig=-(r5 - m5.to_numpy()[:,None]); vol=R.rolling(20).std(); sig=sig/(vol*np.sqrt(5))
disp=R.sub(m,axis=0).rolling(5).std().mean(axis=1); med=disp.rolling(60).median(); gate=(0.5+(disp>med).astype(float)*0.5).to_numpy(); sig=sig*gate[:,None]
fwd=P.pct_change().shift(-1); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,q in [('full',a),('recent756',a.tail(756)),('recent252',a.tail(252))]: print(label,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
for h in [3,5,10,20]:
 y=P.pct_change(h).shift(-h); rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'IC',np.nanmean(rr),'n',len(rr))
print('coverage',sig.notna().sum(axis=1).mean()/15); print('rank turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig.to_csv('scripts/miner_2_20310310_residual_reversal_signal.csv'); a.to_csv('scripts/miner_2_20310310_residual_reversal_ic.csv')
