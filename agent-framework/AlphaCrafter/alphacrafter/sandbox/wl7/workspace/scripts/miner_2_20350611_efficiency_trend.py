import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-06-10'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cut,'close'].astype(float)
common=sorted(set.intersection(*[set(x.index) for x in px.values()])); rec=[]; sigout=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  i=p.index.get_loc(dt)
  if i<45 or i+20>=len(p): continue
  r40=p.iloc[i]/p.iloc[i-40]-1; rets=p.pct_change().iloc[i-39:i+1]; path=rets.abs().sum()
  if path<=0: continue
  vals[s]=r40*(r40/path); fw[s]={h:p.iloc[i+h]/p.iloc[i]-1 for h in [1,5,10,20]}
 if len(vals)<8: continue
 a=np.array(list(vals.values())); med=np.median(a); mad=np.median(np.abs(a-med))+1e-8
 f={s:(v-med)/mad for s,v in vals.items()}
 for h in [1,5,10,20]: rec.append((dt,h,spearmanr([f[s] for s in f],[fw[s][h] for s in f]).statistic,len(f)))
 for s in f: sigout.append({'date':dt.date().isoformat(),'symbol':s,'signal':f[s]})
r=pd.DataFrame(rec,columns=['date','h','ic','n']); print('factor=efficiency_trend cut',cut.date(),'dates',r.date.nunique(),'assets',len(U))
for h in [1,5,10,20]:
 x=r[r.h==h].ic.dropna(); print('H%d dates %d avgN %.2f IC %.6f ICIR %.6f hit %.3f coverage %.3f'%(h,len(x),r[r.h==h].n.mean(),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),(x>0).mean(),r[r.h==h].n.sum()/(len(x)*15)))
for w in [252,756,1260]:
 x=r[r.h==20].tail(w).ic.dropna(); print('RECENT',w,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'dates',len(x))
pd.DataFrame(sigout).to_csv('scripts/miner_2_20350611_efficiency_trend_signal.csv',index=False)
