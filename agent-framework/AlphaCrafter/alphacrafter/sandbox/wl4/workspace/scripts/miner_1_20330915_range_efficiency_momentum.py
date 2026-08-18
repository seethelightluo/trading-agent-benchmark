import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr

files=glob.glob('../persistent/stock_data/*.csv')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); prices[s]=x.set_index('date').close
px=pd.DataFrame(prices).sort_index()
# range-efficiency trend: directional displacement / total path, rewarded by medium momentum
r=px.pct_change()
disp=px/px.shift(30)-1
path=r.abs().rolling(30).sum()
eff=(disp.abs()/path).replace([np.inf,-np.inf],np.nan)
factor=(disp*eff).shift(1) # lag one completed session
fwd=px.shift(-10)/px-1
rows=[]
for d in px.index:
 a=factor.loc[d]; b=fwd.loc[d]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  rows.append((d,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
out=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(out),'avg_n',out.n.mean(),'coverage',out.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1), (out.ic>0).mean()))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [260,520,780]:
 q=out.tail(n); print('recent',n,'dates',len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
for h in [1,5,10,20,30]:
 fw=px.shift(-h)/px-1; rr=[]
 for d in px.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'%.8f'%np.nanmean(rr),'n',len(rr))
# artifact
sig=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
sig.to_csv('scripts/artifacts/miner_1_20330915_range_efficiency_momentum_signal.csv',index=False)
