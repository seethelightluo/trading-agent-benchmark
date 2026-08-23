import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); x=x.set_index('date').sort_index()
 D[s]=x.close.astype(float)
# cross-sectional volatility-gated short reversal: negative 3d return/20d vol, active only when abs move exceeds 1 std
rows=[]
for dt in sorted(set.intersection(*[set(x.index) for x in D.values()])):
 if dt>pd.Timestamp('2028-10-04'): continue
 vals={}; fwd={}
 for s,x in D.items():
  z=x.loc[:dt]
  if len(z)<25: continue
  r=np.log(z.iloc[-1]/z.iloc[-4]); vol=np.log(z).diff().rolling(20).std().iloc[-1]
  if pd.isna(vol) or vol<=0 or len(z)<2: continue
  # standardized move, gated shock; cap prevents extreme crypto domination
  score=-r/vol
  vals[s]=np.clip(score, -3, 3) if abs(r)/vol>=1.0 else 0.0
  nxt=x[x.index>dt]
  if len(nxt): fwd[s]=nxt.iloc[0]/z.iloc[-1]-1
 if len(vals)>=8:
  common=sorted(set(vals)&set(fwd)); a=[vals[s] for s in common]; b=[fwd[s] for s in common]
  if np.std(a)>0 and np.std(b)>0: rows.append((dt,len(common),spearmanr(a,b).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15,'turnover_proxy',np.nan)
print('daily IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for h in [1,5,10]:
 out=[]
 for dt in r.index:
  vals={}; ret={}
  for s,x in D.items():
   z=x.loc[:dt]
   if len(z)<25: continue
   rr=np.log(z.iloc[-1]/z.iloc[-4]); v=np.log(z).diff().rolling(20).std().iloc[-1]
   if pd.isna(v) or v<=0: continue
   vals[s]=np.clip(-rr/v,-3,3) if abs(rr)/v>=1 else 0
   fut=x[x.index>dt]
   if len(fut)>=h: ret[s]=fut.iloc[h-1]/z.iloc[-1]-1
  c=sorted(set(vals)&set(ret))
  if len(c)>=8 and np.std([vals[s] for s in c])>0: out.append(spearmanr([vals[s] for s in c],[ret[s] for s in c]).statistic)
 print('h',h,'dates',len(out),'IC %.6f ICIR %.6f'%(np.mean(out),np.mean(out)/np.std(out,ddof=1)))
