import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2033-10-27'); px={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); px[s]=d.sort_values('date').set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill().loc[:cut]; ret=P.pct_change(20); vol=P.pct_change().rolling(20).std()*np.sqrt(20)
f=(ret.sub(ret.median(axis=1),axis=0)/vol).shift(1); fr=P.shift(-10)/P-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for label,q in [('all',r),('recent520',r.tail(520)),('recent260',r.tail(260)),('recent120',r.tail(120))]:
 ic=q.ic.mean(); sd=q.ic.std(ddof=1); print(label,'dates',len(q),'avgN',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(ic,ic/sd*np.sqrt(252), (q.ic>0).mean()))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True)-f.shift(1).rank(axis=1,pct=True)).abs().mean().mean(),4),'period',r.index.min(),r.index.max())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_1_20331027_risk_adjusted_relative_momentum_signal.csv',index=False)
