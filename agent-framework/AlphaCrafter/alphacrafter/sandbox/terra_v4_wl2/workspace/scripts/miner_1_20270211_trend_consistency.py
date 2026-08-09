import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U if os.path.exists(f'../persistent/stock_data/{s}.csv')}
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); f=(r>0).rolling(20).mean(); target=r.shift(-1)
q=[];ns=[];turn=[];prev=None
for d in f.index:
 z=pd.concat([f.loc[d],target.loc[d]],axis=1).dropna()
 if len(z)>=8:
  q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z)); rr=z.iloc[:,0].rank(pct=True)
  if prev is not None:turn.append(abs(rr-prev).mean())
  prev=rr
q=np.array(q); print('dates',len(q),'avg_n',np.mean(ns),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1),'hit',np.mean(q>0),'turn',np.mean(turn),'coverage',f.notna().sum().sum()/f.size)
for h in [5,10]:
 target=p.pct_change(h).shift(-h); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],target.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(h,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
for n,m in [('2020-22',f.index<'2023'),('2023-24',(f.index>='2023')&(f.index<'2025')),('2025+',f.index>='2025')]:
 a=[]
 for d in f.index[m]:
  z=pd.concat([f.loc[d],target.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(n,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('../persistent/factor_signals_miner_1_20270211_trend_consistency.csv')
