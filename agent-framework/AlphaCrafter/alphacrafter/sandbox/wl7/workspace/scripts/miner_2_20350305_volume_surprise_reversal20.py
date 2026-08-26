import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x.date); x=x.sort_values('date').reset_index(drop=True); c=x.close; v=x.volume
 for i in range(25,len(x)-20):
  med=v.iloc[i-25:i].median(); vs=np.clip(v.iloc[i-5:i+1].mean()/med,0.5,2) if med>0 else 1
  rows.append((x.date.iloc[i],s,-(c.iloc[i]/c.iloc[i-20]-1)*vs,c.iloc[i+20]/c.iloc[i]-1))
a=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']); out=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1: out.append((d,spearmanr(g.factor,g.fwd).statistic,len(g)))
r=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print('dates',len(r),'avgN',r.n.mean(),'minN',r.n.min())
for name,z in [('all',r),('180',r.tail(180)),('500',r.tail(500))]: print(name,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
print('coverage',a.factor.notna().mean()); a.to_csv('scripts/miner_2_20350305_volume_surprise_reversal20_signal.csv',index=False)
