import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2030-01-09');px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna();r=p.pct_change();r5=p.pct_change(5);rel=r5.sub(r5.median(axis=1),axis=0);vol=r.rolling(20,min_periods=15).std()*np.sqrt(252);breadth=(r5>0).mean(axis=1);stress=(breadth.sub(.5).abs()*2).clip(0,1);s=(-rel/vol).mul(1+.5*stress,axis=0)
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [3,5,10,20]:
 x=[];n=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(q.f.corr(q.y,method='spearman'));n.append(len(q))
 x=pd.Series(x);print('TEST',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(n)/15),4))
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',s.notna().mean().mean())
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20300110_stress_reversal_signal.csv',index=False);print('artifact rows',len(out))
