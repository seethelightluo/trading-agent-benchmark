import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date)
 a[s]=d.set_index('date')[['close','volume']].astype(float)
p=pd.DataFrame({s:a[s].close for s in U}).sort_index(); v=pd.DataFrame({s:a[s].volume for s in U}).reindex(p.index)
r=p.pct_change(); ics=[]; ns=[]; signals=[]
for i in range(65,len(p)-10):
 # Prior-day observable 5-day reversal, amplified by unusual trading activity.
 ret5=r.iloc[i-4:i+1].sum(); vol_ratio=v.iloc[i]/(v.iloc[i-20:i].mean()+1e-12)
 f=-ret5*np.log1p(vol_ratio.clip(lower=0))
 y=p.iloc[i+10]/p.iloc[i]-1; z=pd.concat([f,y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); signals.append(f)
x=np.array(ics)
print('factor=volume_confirmed_reversal_5d_h10 dates=%d avgN=%.2f coverage=%.4f IC=%.8f ICIR=%.8f hit=%.4f'%(len(x),np.mean(ns),np.mean(ns)/15,np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1)*np.sqrt(len(x)),np.mean(x>0)))
for n in [250,500,750,1000]:
 q=x[-min(n,len(x)):]; print('recent',len(q),'IC=%.8f ICIR=%.8f hit=%.4f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),np.mean(q>0)))
# rank turnover measured at the 10-day rebalance cadence
turn=[]
for j in range(10,len(signals),10):
 u=signals[j].rank(pct=True); w=signals[j-10].rank(pct=True); turn.append(np.mean(abs(u-w)))
print('turnover_proxy=%.6f'%np.mean(turn))
