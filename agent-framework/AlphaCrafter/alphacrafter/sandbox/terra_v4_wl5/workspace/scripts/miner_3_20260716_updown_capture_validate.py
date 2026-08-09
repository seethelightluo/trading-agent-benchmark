import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index()
r=p.pct_change(); market=r.median(axis=1)
def factor(win):
 out=pd.DataFrame(index=r.index,columns=U,dtype=float)
 for i in range(win,len(r)):
  h=r.iloc[i-win:i]; bm=market.iloc[i-win:i]; up=bm>0; dn=bm<0
  if up.sum()>=4 and dn.sum()>=4:
   out.iloc[i]=h.where(np.repeat(up.to_numpy()[:,None],len(U),1)).mean()-h.where(np.repeat(dn.to_numpy()[:,None],len(U),1)).mean()
 return out
def run(f,h):
 ic=[]; ns=[]; ranks=[]
 for i in range(len(r)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('fwd')],axis=1).dropna()
  if len(q)>=8:
   ic.append(spearmanr(q.iloc[:,0],q.fwd).statistic); ns.append(len(q)); ranks.append(q.iloc[:,0].rank(pct=True))
 x=np.asarray(ic); turn=np.nanmean([np.abs(ranks[j]-ranks[j-1]).mean() for j in range(1,len(ranks))]) if len(ranks)>1 else np.nan
 print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'rank_turnover',round(turn,4))
 print('regimes', {f'{a}-{b}':round(pd.Series(x,index=[d for d in f.index if d in []]).mean(),5) for a,b in []})
for w in [20,60]:
 f=factor(w); print('WINDOW',w)
 for h in [1,5,10,20]: run(f,h)
 # regime means of 1d IC
 vals=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8: vals.append((f.index[i],spearmanr(q.iloc[:,0],q.y).statistic))
 z=pd.Series(dict(vals)); print('REGIME_IC', {str(y):round(z[z.index.year==y].mean(),5) for y in sorted(z.index.year.unique())})
