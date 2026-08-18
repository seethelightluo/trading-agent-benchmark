import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); dc='date' if 'date' in d else d.columns[0]; cc='close' if 'close' in d else 'Close'; d[dc]=pd.to_datetime(d[dc]); px[s]=d.set_index(dc)[cc].astype(float).sort_index()
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); v=r.rolling(20,min_periods=15).std(); long=r.rolling(60,min_periods=40).std(); F=-(v/long); y=P.shift(-10)/P-1; rows=[]
for dt in P.index:
 z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,x in [('all',q),('780',q.tail(780)),('520',q.tail(520)),('260',q.tail(260)),('120',q.tail(120))]:
 m=x.ic.mean(); sd=x.ic.std(ddof=1); print(label,len(x),round(x.n.mean(),2),round(m,6),round(m/sd*np.sqrt(252),6),round((x.ic>0).mean(),4))
print('coverage',round(q.n.mean()/15,4),'period',q.index.min(),q.index.max())
os.makedirs('scripts/artifacts',exist_ok=True); F.to_csv('scripts/artifacts/miner_1_20350104_lowvol_compression_signal.csv')
