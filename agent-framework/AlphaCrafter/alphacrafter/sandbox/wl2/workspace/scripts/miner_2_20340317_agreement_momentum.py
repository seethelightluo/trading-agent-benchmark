import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    f='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(f,parse_dates=['date']).set_index('date')
    px[s]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); P=P.loc[P.index<='2034-03-16']; r=P.pct_change()
# interpretable factor: risk-adjusted medium momentum, strengthened by agreement of short/medium/long trends
mom20=P.pct_change(20); vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
agree=(np.sign(P.pct_change(5))+np.sign(P.pct_change(20))+np.sign(P.pct_change(60)))/3
F=(mom20/vol20)*agree
# factor observable at t; forward return t+1
y=r.shift(-1)
rows=[]
for dt in P.index:
    x=F.loc[dt]; z=y.loc[dt]; q=pd.concat([x,z],axis=1).dropna()
    if len(q)>=8:
        ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
        rows.append((dt,ic,len(q)))
D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('range',D.index.min(),D.index.max(),'dates',len(D),'avg_n',D.n.mean(),'coverage',D.n.sum()/(len(D)*15))
for h in [1,5,10,20]:
    yy=P.pct_change(h).shift(-h)
    rr=[]
    for dt in P.index:
      q=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).dropna()
      if len(q)>=8: rr.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
    a=np.array(rr); print(h,'IC',a.mean(),'ICIR_daily',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'obs',len(a))
for start in ['2020-01-01','2024-01-01','2028-01-01','2031-01-01','2033-01-01']:
 a=D[D.index>=start].ic
 print(start,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
# signal artifact
out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340317_agreement_momentum_signal.csv',index=False)
