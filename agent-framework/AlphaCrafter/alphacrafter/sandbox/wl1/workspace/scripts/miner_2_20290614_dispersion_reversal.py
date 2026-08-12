import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-06-13')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U};idx=sorted(set().union(*[set(x.index) for x in P.values()]));px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill();r=px.pct_change()
cs=r.std(axis=1).rolling(20,min_periods=15).mean(); base=cs.rolling(120,min_periods=60).median(); gate=(1+cs.div(base).replace([np.inf,-np.inf],np.nan)); gate=pd.DataFrame(np.repeat(gate.to_numpy()[:,None],len(U),axis=1),index=px.index,columns=U)
raw=-px.pct_change(5).div(r.rolling(20,min_periods=15).std()*np.sqrt(252)+1e-8);f=(raw*gate).shift(1)
print('factor dispersion_gated_reversal_risk_5_20 universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [1,5,10,20]:
 I=[];N=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q))
 a=np.asarray(I);print(h,'dates',len(a),'avgN',round(np.mean(N),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
 if h==10:
  for label,lo in [('2020-25','2020-01-01'),('2026+','2026-01-01'),('2028+','2028-01-01'),('2029YTD','2029-01-01')]:
   z=[]
   for i in range(len(px)-h):
    if px.index[i]<pd.Timestamp(lo):continue
    q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
    if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:z.append(spearmanr(q.f,q.y).statistic)
   z=np.asarray(z);print(' ',label,len(z),round(z.mean(),6),round(z.mean()/(z.std(ddof=1)+1e-12)*np.sqrt(252),6))
rr=f.rank(axis=1,pct=True);print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(rr.diff().abs().mean(axis=1).mean(),6));out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20290614_dispersion_reversal_signal.csv',index=False)
