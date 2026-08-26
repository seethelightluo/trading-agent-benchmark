import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-03-19')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index().loc[:cut]
r=p.pct_change(); f=(-r.rolling(10).sum()/(r.rolling(60).std()*np.sqrt(10)+1e-8)).shift(1)
print('assets',len(assets),'dates',len(p),'period',p.index.min().date(),p.index.max().date())
for h in [1,5,10,20]:
 v=[]; ns=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i],(p.iloc[i+h+1]/p.iloc[i+1]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
   v.append(spearmanr(z.iloc[:,0],z.y).statistic);ns.append(len(z))
 s=pd.Series(v); print('h',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),3),'dates',len(s),'avgN',round(np.mean(ns),2))
rank=f.rank(axis=1,pct=True)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna().to_csv('scripts/miner_2_20340320_drawdown_signal.csv',index=False)
