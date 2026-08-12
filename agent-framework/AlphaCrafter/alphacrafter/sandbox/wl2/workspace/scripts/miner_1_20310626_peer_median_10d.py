import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; q={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);q[a]=d.set_index('date').close
p=pd.DataFrame(q).sort_index();r=p.pct_change(); peer=r.rolling(10).sum().sub(r.rolling(10).sum().median(axis=1),axis=0).shift(1); fw=p.shift(-10).div(p)-1
z=[];ds=[];ins=[]
for d in peer.index:
 ok=peer.loc[d].notna()&fw.loc[d].notna()
 if ok.sum()>=8:z.append(spearmanr(peer.loc[d][ok],fw.loc[d][ok]).statistic);ds.append(d);ins.append(ok.sum())
z=pd.Series(z,index=ds);print('dates',len(z),'avg_inst',np.mean(ins),'universe',15);print('IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
for n,a,b in [('2020-22','2020','2023'),('2023-25','2023','2026'),('2026-31','2026','2032')]:
 x=z[(z.index>=a)&(z.index<b)];print(n,len(x),x.mean(),x.mean()/x.std())
peer.to_csv('scripts/miner_1_20310626_peer_median_10d_signal.csv',index_label='date')
