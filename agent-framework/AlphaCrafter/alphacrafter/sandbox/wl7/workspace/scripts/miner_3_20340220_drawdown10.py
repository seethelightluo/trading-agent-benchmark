import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-02-19')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index()
p=p[p.index<=cut]; r=p.pct_change()
# Candidate: lagged 10-session volatility-normalized drawdown reversal.
f=(-r.rolling(10).sum()/(r.rolling(60).std()*np.sqrt(10)+1e-8)).shift(1)
rows=[]
for h in [1,5,10,20]:
 v=[]; ns=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i],(p.iloc[i+h+1]/p.iloc[i+1]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
   v.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z))
 s=pd.Series(v); rows.append((h,s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(ns)))
for x in rows: print('h %d IC %.6f ICIR %.6f hit %.3f dates %d avgN %.2f'%x)
rank=f.rank(axis=1,pct=True); print('coverage %.4f turnover %.4f period %s to %s assets %d'%(f.notna().mean().mean(),rank.diff().abs().mean(axis=1).mean(),p.index.min().date(),p.index.max().date(),len(assets)))
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna().to_csv('scripts/miner_3_20340220_drawdown10_signal.csv',index=False)
