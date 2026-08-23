import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
root='../persistent/stock_data'; assets=[f[:-4] for f in os.listdir(root) if f.endswith('.csv')]
px={}
for a in assets:
 d=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').sort_index(); px[a]=d.close.replace(0,np.nan)
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# Reversal amplified when VIX is falling (risk stabilisation); continuation when VIX rising.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
v=v.reindex(close.index).ffill(); regime=np.where(v.shift(1).pct_change(5)<0,1.0,-1.0)
f=(-close.pct_change(5)*regime[:,None]).shift(1).replace([np.inf,-np.inf],np.nan)
for h in [1,3,5,10]:
 fr=close.shift(-h)/close-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 s=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); print(h,'dates',len(s),'avg_n',round(s.n.mean(),2),'IC',round(s.ic.mean(),6),'ICIR',round(s.ic.mean()/s.ic.std(ddof=1),6),'hit',round((s.ic>0).mean(),4))
 for w in [180,360]:
  q=s.tail(w); print(' recent',w,round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),len(q))
print('coverage',round(f.notna().sum().sum()/f.size,4),'assets',len(assets)); f.to_csv('scripts/miner_2_20281228_vix_trend_reversal_signal.csv',index_label='date')
