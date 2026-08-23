import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); data[s]=d.set_index('date').sort_index()
# candidate: lagged path-efficiency momentum = 20d return / sum abs daily returns, multiplied by sign persistence
frames=[]
for s,d in data.items():
 r=d.close.pct_change()
 mom=d.close/d.close.shift(20)-1
 eff=mom/(r.abs().rolling(20).sum()+1e-12)
 pos=(r>0).rolling(20).mean()
 sig=(eff*(0.5+pos)).shift(1)
 fwd=d.close.shift(-10)/d.close-1
 z=pd.DataFrame({'date':d.index,'s':s,'sig':sig.values,'fwd':fwd.values}).dropna(); frames.append(z)
x=pd.concat(frames)
ics=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: ics.append((dt,spearmanr(g.sig,g.fwd).statistic,len(g)))
a=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',len(x)/sum(len(d) for d in data.values()))
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean(),'turnover unavailable')
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=a.loc[lo:hi].ic;print(lo, len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# decay
for h in [1,5,10,20]:
 fs=[]
 for s,d in data.items():
  r=d.close.pct_change(); mom=d.close/d.close.shift(20)-1; eff=mom/(r.abs().rolling(20).sum()+1e-12); pos=(r>0).rolling(20).mean(); sig=eff*(.5+pos)
  fs.append(pd.DataFrame({'date':d.index,'s':s,'sig':sig.shift(1).values,'fwd':(d.close.shift(-h)/d.close-1).values}).dropna())
 xx=pd.concat(fs); ii=[]
 for dt,g in xx.groupby('date'):
  if len(g)>=8: ii.append(spearmanr(g.sig,g.fwd).statistic)
 print('decay',h,np.nanmean(ii),len(ii))
# signal artifact
out=x.pivot(index='date',columns='s',values='sig');out.to_csv('scripts/miner_1_20270615_path_efficiency_momentum_signal.csv')
