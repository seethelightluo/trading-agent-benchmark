import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-10-25'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
# Efficiency-adjusted momentum: lagged 20d return divided by 40d path length.
path=R.abs().rolling(40,min_periods=20).sum()
F=P.pct_change(20).shift(1).div(path.shift(1)+1e-12)
rows=[]; daily=[]
for i in range(65,len(P)-10):
 z=pd.concat([F.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(ic): rows.append((P.index[i],ic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')['ic']
recent=a.iloc[-252:]; early=a.iloc[:len(a)//3]; mid=a.iloc[len(a)//3:2*len(a)//3]
# rank turnover every 10 sessions, using signal snapshots
snap=F.iloc[65::10].rank(pct=True,axis=1); turn=snap.diff().abs().mean(axis=1).mean()
print('cutoff',cutoff.date(),'rows',len(P),'instruments',len(D),'ic_dates',len(a),'avg_n',round(np.mean([x[2] for x in rows]),2))
print('ic',round(a.mean(),6),'icir',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4),'recent_ic',round(recent.mean(),6),'recent_icir',round(recent.mean()/recent.std(),6),'early_mid_recent',*[round(x.mean(),6) for x in [early,mid,recent]],'coverage',round(np.mean([x[2]/15 for x in rows]),4),'rank_turnover',round(turn,5))
F.to_csv('scripts/miner_3_20351026_efficiency_momentum20_signal.csv')
