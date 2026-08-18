import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
base='../persistent/stock_data'
syms=sorted([x[:-4] for x in os.listdir(base) if x.endswith('.csv')])
px={}
for s in syms:
 d=pd.read_csv(os.path.join(base,s+'.csv'))
 d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index()
# shock reversal: contrarian recent return amplified when short vol is elevated vs long vol
r=P.pct_change()
short=r.rolling(5,min_periods=5).std(); long=r.rolling(60,min_periods=60).std()
f=-(P.pct_change(3))*((short/(long+1e-12)).clip(0,5))
# signal known at t, forward 10 sessions from t+1 through t+10
fr=P.shift(-10)/P.shift(-1)-1
rows=[]
for dt in P.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
out=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('universe',len(syms),'dates',len(out),'meanN',out.n.mean(),'coverage',out.n.mean()/len(syms))
print('IC',out.ic.mean(),'ICIR',out.ic.mean()/out.ic.std(ddof=1),'hit',(out.ic>0).mean())
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 q=out.loc[a:b].ic
 if len(q): print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean())
# save signal artifact only if later accepted
print('recent',out.tail(60).ic.mean(),out.tail(60).ic.mean()/out.tail(60).ic.std(ddof=1))
# turnover rank changes, median spearman consecutive signals
rank=f.rank(axis=1,pct=True); rr=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8: rr.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('turnover_proxy',np.nanmean(rr))
