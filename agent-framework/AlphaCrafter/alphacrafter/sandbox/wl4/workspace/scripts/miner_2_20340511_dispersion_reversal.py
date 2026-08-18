import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
trad=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in trad:
 f=f'{base}/{s}.csv'
 if not os.path.exists(f): f=f'../persistent/index_data/{s}.csv'
 d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# cross-sectional daily dispersion, smoothed and standardized using trailing history only
disp=r.std(axis=1).rolling(20,min_periods=15).mean()
mu=disp.rolling(120,min_periods=60).mean(); sd=disp.rolling(120,min_periods=60).std()
reg=((disp-mu)/sd).clip(-1.5,1.5)
# reversal, scaled more in high dispersion, lagged
f=-(p/p.shift(20)-1)/r.rolling(40,min_periods=25).std()/np.sqrt(40)
f=f.mul((1+0.7*reg).shift(1),axis=0)
f=f.shift(1)
rows=[]
for i in range(len(p)-10):
 dt=p.index[i]
 if dt < pd.Timestamp('2026-07-20'): continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1
 ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15))
print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'turnover',np.nan)
for w in [120,260,520,780]:
 q=z.tail(w); print('window',w,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
# rank turnover proxy
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna(); print('rank_turnover',turn.mean())
out='scripts/artifacts'; os.makedirs(out,exist_ok=True)
sig=f.loc[z.index].reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); sig.to_csv(out+'/miner_2_20340511_dispersion_reversal_signal.csv',index=False)
z.reset_index().to_csv(out+'/miner_2_20340511_dispersion_reversal_ic.csv',index=False)
