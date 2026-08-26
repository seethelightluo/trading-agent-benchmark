import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-11-27')
files=glob.glob('../persistent/stock_data/*.csv')
px={}
for f in files:
 s=f.split('/')[-1][:-4]; d=pd.read_csv(f); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date'); px[s]=d.close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# conditional short reversal: reverse 5d return, only when medium trend is positive; risk adjusted
f=-(p.pct_change(5))/(r.rolling(20).std()*np.sqrt(252))
trend=p.pct_change(60)
f=f.where(trend>0, 0.0).shift(1)
rows=[]
for dt in p.index:
 if dt not in f.index: continue
 vals=f.loc[dt]; rr=p.pct_change(10).shift(-10).loc[dt]
 z=pd.concat([vals,rr],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(x),'avgN',x.n.mean(),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 ff=-(p.pct_change(5))/(r.rolling(20).std()*np.sqrt(252)); ff=ff.where(trend>0,0).shift(1)
 rr=p.pct_change(h).shift(-h); a=[]
 for dt in p.index:
  z=pd.concat([ff.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
print('recent',x.tail(500).ic.mean(),x.tail(500).ic.mean()/x.tail(500).ic.std(ddof=1))
# signal artifact
f.to_csv('scripts/miner_2_20341127_conditional_reversal_signal.csv')
