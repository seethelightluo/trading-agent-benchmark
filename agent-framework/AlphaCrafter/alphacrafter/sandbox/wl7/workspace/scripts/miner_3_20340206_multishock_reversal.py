import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-02-05'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index();p=p[p.index<=cut];r=p.pct_change()
# Multi-day volatility-shock reversal: lagged 5-session drawdown normalized by 60-session volatility.
f=(-r.rolling(5).sum()/(r.rolling(60).std()*np.sqrt(5)+1e-8)).shift(1);f=f.sub(f.median(axis=1),axis=0)
for h in [5,10,20]:
 v=[];n=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i],(p.iloc[i+h+1]/p.iloc[i+1]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:v.append(spearmanr(z.iloc[:,0],z.y).statistic);n.append(len(z))
 s=pd.Series(v);print('h',h,'IC %.6f ICIR %.6f hit %.3f dates %d avgN %.2f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(n)))
print('coverage %.4f turnover %.4f period %s to %s assets %d'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),p.index.min().date(),p.index.max().date(),len(assets)))
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna().to_csv('scripts/miner_3_20340206_multishock_signal.csv',index=False)
