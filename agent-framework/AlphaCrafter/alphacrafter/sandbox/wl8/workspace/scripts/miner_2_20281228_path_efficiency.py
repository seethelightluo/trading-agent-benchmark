import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
root='../persistent/stock_data'
assets=[f[:-4] for f in os.listdir(root) if f.endswith('.csv')]
px={}
for a in assets:
 d=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 px[a]=d['close'].replace(0,np.nan)
close=pd.DataFrame(px).sort_index()
r=close.pct_change()
# Path efficiency: directional net movement relative to total movement, lagged one day.
net=close/close.shift(20)-1
path=r.rolling(20,min_periods=15).apply(lambda x: np.nansum(np.abs(x)),raw=True)
f=(net/path).shift(1).replace([np.inf,-np.inf],np.nan).clip(-1,1)
# forward close-to-close returns from information date t
out=[]
for h in [1,3,5,10]:
 fr=close.shift(-h)/close-1
 vals=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 s=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print(h,'dates',len(s),'avg_n',round(s.n.mean(),2),'IC',round(s.ic.mean(),6),'ICIR',round(s.ic.mean()/s.ic.std(ddof=1),6),'hit',round((s.ic>0).mean(),4))
 for w in [180,360]:
  q=s.tail(w); print(' recent',w,round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),len(q))
print('coverage',round(f.notna().sum().sum()/f.size,4),'rows',f.notna().sum().sum(),'assets',len(assets))
f.to_csv('scripts/miner_2_20281228_path_efficiency_signal.csv',index_label='date')
