import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-07-22']
r=C.pct_change(); down=r.where(r<0,0)
# tail-loss rebound: recent loss severity relative to downside risk, with a mild long-term trend gate
loss=-(C/C.shift(5)-1)
downvol=down.rolling(40,min_periods=25).std()*np.sqrt(40)
trend=(C/C.shift(60)-1)
F=(loss/downvol)*(1+0.5*np.maximum(-trend,0))
F=F.sub(F.median(axis=1),axis=0).shift(1)
y=C.shift(-20)/C-1
A=[]; ds=[]; ns=[]
for d in F.index:
 q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8:A.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(d);ns.append(len(q))
a=np.array(A);ds=np.array(ds)
print('tail_loss_rebound IC %.8f ICIR %.8f dates %d avgN %.2f hit %.4f coverage %.4f'%(a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),len(a),np.mean(ns),np.mean(a>0),F.notna().sum().sum()/(15*len(F))))
for st,en in [('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-07-22')]:
 q=a[(ds>=pd.Timestamp(st))&(ds<=pd.Timestamp(en))]; print('regime',st,len(q),q.mean() if len(q) else None)
for h in [1,5,10,20]:
 yy=C.shift(-h)/C-1; aa=[]
 for d in F.index:
  q=pd.concat([F.loc[d],yy.loc[d]],axis=1).dropna()
  if len(q)>=8:aa.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 aa=np.array(aa);print('H',h,'IC %.8f ICIR %.8f dates %d'%(aa.mean(),aa.mean()/aa.std(ddof=1)*np.sqrt(252),len(aa)))
F.to_csv('scripts/miner_3_20350723_tail_loss_rebound_signal.csv',index_label='date')
