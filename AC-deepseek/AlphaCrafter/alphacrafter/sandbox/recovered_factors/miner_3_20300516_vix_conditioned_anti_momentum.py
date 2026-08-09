import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index().astype(float)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# one candidate: anti-momentum is activated in elevated VIX (above trailing 120d median)
base=-(p/p.shift(20)-1); gate=v>v.rolling(120,min_periods=60).median(); sig=base.where(gate)
print('assets',p.shape[1],'dates',p.index.min().date(),p.index.max().date(),'gate%',gate.mean())
for h in [1,3,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[];ns=[]; prev=None;turn=[]
 for d in p.index:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  r=sig.loc[d].rank(pct=True); 
  if prev is not None: turn.append(pd.concat([prev,r],axis=1).dropna().diff(axis=1).iloc[:,-1].abs().mean())
  prev=r
 a=np.array(vals);print('H',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),3),'turn',round(np.nanmean(turn),4))
 for name,mask in [('24-27',(p.index>='2024')&(p.index<='2027-12-31')),('28+',p.index>='2028'),('last120',np.arange(len(p))>=len(p)-120)]:
  aa=[]
  for d in p.index[mask]:
   z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
   if len(z)>=8:aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  aa=np.array(aa);print(name,len(aa),round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6) if len(aa)>1 else np.nan)
