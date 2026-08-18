import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def main():
 syms=U; px={}
 for s in syms:
  d=get_stock_daily_data(s,days=5000)
  if d is not None: px[s]=d.set_index('date')['close'].astype(float)
 p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
 b=r[['XAU','US10Y','CN10Y']].mean(axis=1); br=b.rolling(60).sum(); bz=br/(b.rolling(60).std()*np.sqrt(60))
 # residualized reversal, strength-weighted continuously; all calculations lagged by one day
 out={}
 for s in syms:
  beta=r[s].rolling(90).cov(b)/b.rolling(90).var()
  res=r[s]-beta*b
  rev=-res.rolling(30).sum()/(res.rolling(60).std()*np.sqrt(60))
  out[s]=rev
 f=pd.DataFrame(out).mul(np.maximum(bz,0),axis=0).shift(1)
 rows=[]
 for i in range(180,len(p)-40):
  x=f.iloc[i].dropna()
  if len(x)<8: continue
  row={'date':p.index[i],'n':len(x)}
  for h in [5,10,20,40]: row['ic'+str(h)]=x.corr(p.iloc[i+h].div(p.iloc[i]).sub(1).reindex(x.index),method='spearman')
  rows.append(row)
 q=pd.DataFrame(rows); print('dates',len(q),'mean_n',q.n.mean(),'period',q.date.min(),q.date.max())
 for h in [5,10,20,40]:
  x=q['ic'+str(h)].dropna();print(h,'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
 print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
 q.to_csv('../persistent/miner_1_20350216_continuous_stress_residual_ic.csv',index=False);f.to_csv('../persistent/miner_1_20350216_continuous_stress_residual_signal.csv')
if __name__=='__main__':main()
