import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-10-03'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']); px[s]=d[d.date<=cut].set_index('date').close.sort_index()
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std(); m=p.pct_change(90); sh=p.pct_change(20); agree=np.where(np.sign(m)==np.sign(sh),1,.45); raw=m/(v*np.sqrt(90)).clip(lower=1e-5)*agree; sig=raw.rank(axis=1,pct=True)
def run(h,a=None,b=None):
 z=[];dt=[]
 for i in range(90,len(p)-h):
  if a and not(pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b)):continue
  q=pd.DataFrame({'f':sig.iloc[i],'y':p.iloc[i+h]/p.iloc[i]-1}).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y,method='spearman'));dt.append(p.index[i])
 x=pd.Series(z,index=dt).dropna();return len(x),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
print('rows',len(p),'range',p.index.min().date(),p.index.max().date())
for h in [5,10,20]:print('ALL',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-10-03')]:print('REG10',a,b,run(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20291004_trend_persistence_signal.csv',index=False);print('artifact_rows',len(out))