import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
eq=assets[:8]
def load(sym, macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+sym+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].rename(sym)
 return d
px=pd.concat([load(s) for s in assets],axis=1)
vix=load('VIX',True).reindex(px.index).ffill()
# factor at date t, using only t and prior: relative 20d momentum, conditionally amplified in VIX shocks
r20=px/px.shift(20)-1
bench=r20[eq].mean(axis=1)
rel=r20.sub(bench,axis=0)
vshock=vix.pct_change(5)
# interpretable: defensive relative strength receives 1.5x weight during VIX shock, 0.75x otherwise
state=np.where(vshock>0.10,1.5,0.75)
f=rel.div(px.pct_change().rolling(20).std()*np.sqrt(252)).mul(state,axis=0)
rows=[]
for h in [5,10,20]:
 vals=[]
 for i in range(len(px)-h):
  # factor lag means signal from prior completed day for forward after t
  x=f.iloc[i]; y=px.iloc[i+h]/px.iloc[i]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append((px.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 ic=q.ic.mean(); icir=ic/q.ic.std(ddof=1); hit=(q.ic>0).mean()
 # turnover rank signal successive dates
 ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
 print(h,'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic,icir,hit,turnover))
 for label,mask in [('2026+',q.index>='2026-01-01'),('2027+',q.index>='2027-01-01'),('2028YTD',q.index>='2028-01-01')]:
  z=q[mask]; print(label,len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
 # signal artifact
 if h==20: f.to_csv('scripts/miner_1_20280420_vix_shock_defensive_relative_strength_signal.csv',index_label='date')
