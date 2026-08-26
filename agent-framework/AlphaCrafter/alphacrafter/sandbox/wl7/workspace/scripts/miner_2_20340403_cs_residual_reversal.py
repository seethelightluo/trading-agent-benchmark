import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-04-02')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index().loc[:cut]
r=px.pct_change()
# Cross-sectional residual short-term reversal: reverse each asset's 5d move relative to same-day median, risk scaled.
cs=r.rolling(5).sum().sub(r.rolling(5).sum().median(axis=1),axis=0)
vol=r.rolling(40).std()*np.sqrt(5)
f=(-cs/(vol+1e-8)).shift(1)
print('assets',len(assets),'dates',len(px),'period',px.index.min().date(),px.index.max().date())
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(px)-h-1):
  z=pd.concat([f.iloc[i],(px.iloc[i+h+1]/px.iloc[i+1]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z))
 s=pd.Series(vals)
 print('h',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),3),'dates',len(s),'avgN',round(np.mean(ns),2))
rank=f.rank(axis=1,pct=True)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna().to_csv('scripts/miner_2_20340403_cs_residual_reversal_signal.csv',index=False)
