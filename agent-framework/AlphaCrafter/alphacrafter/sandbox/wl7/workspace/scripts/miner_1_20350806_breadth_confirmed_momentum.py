import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-08-05']
r=C.pct_change(); m20=C/C.shift(20)-1; v40=r.rolling(40,min_periods=25).std()*np.sqrt(40)
breadth=(m20>0).mean(axis=1); F=(m20/v40).mul(0.5+breadth,axis=0)
F=F.sub(F.median(axis=1),axis=0).shift(1)
def evalf(y):
 A=[];ds=[];ns=[]
 for d in F.index:
  q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:A.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(d);ns.append(len(q))
 return np.array(A),pd.to_datetime(ds),ns
for h in [1,5,10,20]:
 a,ds,ns=evalf(C.shift(-h)/C-1); ir=a.mean()/a.std(ddof=1)*np.sqrt(252)
 print('H%d IC %.8f ICIR %.8f dates %d avgN %.2f hit %.4f coverage %.4f'%(h,a.mean(),ir,len(a),np.mean(ns),np.mean(a>0),F.notna().sum().sum()/(15*len(F))))
a,ds,ns=evalf(C.shift(-10)/C-1)
for st,en in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-12-31'),('2035','2035-08-05')]:
 q=a[(ds>=pd.Timestamp(st))&(ds<=pd.Timestamp(en))]; ir=q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan
 print('regime',st,len(q),'IC',q.mean() if len(q) else None,'ICIR',ir)
turn=F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna(); print('turnover_proxy %.6f'%turn.mean())
F.to_csv('scripts/miner_1_20350806_breadth_confirmed_momentum_signal.csv',index_label='date')
