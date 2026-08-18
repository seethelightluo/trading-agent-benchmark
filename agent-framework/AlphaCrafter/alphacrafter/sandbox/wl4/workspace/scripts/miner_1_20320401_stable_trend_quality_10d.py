import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); a[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(a).sort_index(); r=p.pct_change(); x=[]; ns=[]; sig=[]
for i in range(65,len(p)-10):
 mom=r.iloc[i-19:i+1].sum(); vol=r.iloc[i-19:i+1].std(); trend=r.iloc[i-59:i+1].sum()
 f=trend/(vol+1e-8)+0.5*mom/(vol+1e-8)
 y=p.iloc[i+10]/p.iloc[i]-1; z=pd.concat([f,y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));sig.append(f)
x=np.array(x); ir=x.mean()/x.std(ddof=1)*np.sqrt(len(x))
print('factor=stable_trend_quality_10d_h10 dates=%d avgN=%.2f coverage=%.4f IC=%.8f ICIR=%.8f hit=%.4f'%(len(x),np.mean(ns),np.mean(ns)/15,x.mean(),ir,np.mean(x>0)))
for n in [250,500,750,1000]:
 q=x[-min(n,len(x)):]; print('recent',len(q),'IC=%.8f ICIR=%.8f hit=%.4f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),np.mean(q>0)))
turn=[]
for j in range(10,len(sig),10): turn.append(np.mean(abs(sig[j].rank(pct=True)-sig[j-10].rank(pct=True))))
print('turnover_proxy=%.6f'%np.mean(turn))
