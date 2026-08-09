import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
px=pd.concat(D,axis=1).sort_index().loc[:'2027-03-24']; r=px.pct_change(); ret5=px.pct_change(5); down=r.clip(upper=0).rolling(20,min_periods=12).std(); f=(-ret5/(down*np.sqrt(252)+1e-8)).clip(-10,10)
rows=[]
for i in range(len(px)-5):
 x=f.iloc[i]; y=px.iloc[i+1:i+6].iloc[-1]/px.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(v):rows.append((px.index[i],v,len(z)))
a=np.array([x[1] for x in rows]);print('period',px.index.min(),px.index.max(),'dates',len(a),'avgN',np.mean([x[2] for x in rows]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',f.notna().sum().sum()/f.size)
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
 q=np.array([x[1] for x in rows if str(x[0])[:10]>=lo and str(x[0])[:10]<=hi]); print(name,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame(f.stack(),columns=['signal']);out.index.names=['date','symbol'];out.reset_index().to_csv('scripts/miner_3_20270325_downside_reversal_signal.csv',index=False)
