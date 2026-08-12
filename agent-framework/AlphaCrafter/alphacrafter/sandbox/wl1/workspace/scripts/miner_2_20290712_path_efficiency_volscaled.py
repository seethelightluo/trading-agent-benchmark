import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-07-11')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Path-efficiency trend: net 40d return relative to total absolute path, penalized by realized volatility; lag one session.
net=px.pct_change(40); path=r.abs().rolling(40,min_periods=30).sum(); vol=r.rolling(40,min_periods=30).std()*np.sqrt(252)
f=(net/(path+1e-8)/(vol+1e-8)).shift(1)
print('factor path_efficiency_volscaled_40 universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
def calc(lo=None,h=10):
 I=[];N=[]
 for i in range(len(px)-h):
  if lo is not None and px.index[i]<pd.Timestamp(lo): continue
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q))
 a=np.asarray(I); return len(a),np.mean(N),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),(a>0).mean()
for h in [1,5,10,20]:
 z=calc(h=h);print(h,'dates',z[0],'avgN',round(z[1],2),'IC',round(z[2],6),'ICIR',round(z[3],6),'hit',round(z[4],4))
for label,lo in [('2020-25','2020-01-01'),('2026+','2026-01-01'),('2028+','2028-01-01'),('2029YTD','2029-01-01')]:
 z=calc(lo,h=10);print(label,z[0],round(z[2],6),round(z[3],6),round(z[4],4))
rr=f.rank(axis=1,pct=True);print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(rr.diff().abs().mean(axis=1).mean(),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20290712_path_efficiency_volscaled_signal.csv',index=False)
