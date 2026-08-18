import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if not os.path.exists(p): continue
 d=pd.read_csv(p); dc='date' if 'date' in d else d.columns[0]; cc='close' if 'close' in d else 'Close'
 d[dc]=pd.to_datetime(d[dc]); px[s]=d.set_index(dc)[cc].astype(float).sort_index()
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
F=-(P/P.shift(5)-1)/vol; fr=P.shift(-10)/P-1; rows=[]
for dt in P.index:
 z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,x in [('all',q),('recent780',q.tail(780)),('recent520',q.tail(520)),('recent260',q.tail(260)),('recent120',q.tail(120))]:
 m=x.ic.mean(); sd=x.ic.std(ddof=1); print(label,'dates',len(x),'avg_n',round(x.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),6) if sd else None,'hit',round((x.ic>0).mean(),4))
rank=F.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().mean(axis=1).dropna()
print('assets',len(px),'coverage',round(q.n.mean()/len(U),4),'turnover',round(turn.mean(),4),'period',q.index.min(),q.index.max())
out='scripts/artifacts/miner_1_20350104_short_reversal_volscaled_signal.csv'; os.makedirs(os.path.dirname(out),exist_ok=True); F.loc[q.index].to_csv(out)
