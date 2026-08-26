import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-03-17')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index().loc[:cut]
r=p.pct_change()
# Lagged 20-day reversal normalized by 90-day realized risk; intended for 10-day rebalance horizon.
f=(-r.rolling(20).sum()/(r.rolling(90).std()*np.sqrt(20)+1e-8)).shift(1)
for h in [1,5,10,20]:
 vals=[];ns=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i],(p.iloc[i+h+1]/p.iloc[i+1]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.y).statistic);ns.append(len(z))
 s=pd.Series(vals)
 print(f'h {h} IC {s.mean():.6f} ICIR {s.mean()/s.std():.6f} hit {(s>0).mean():.3f} dates {len(s)} avgN {np.mean(ns):.2f}')
rank=f.rank(axis=1,pct=True)
print(f'coverage {f.notna().mean().mean():.4f} turnover {rank.diff().abs().mean(axis=1).mean():.4f} period {p.index.min().date()} to {p.index.max().date()} assets {len(assets)}')
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna().to_csv('scripts/miner_1_20340320_medium_reversal_signal.csv',index=False)
