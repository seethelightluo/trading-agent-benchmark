import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2035-03-29')
px={}
for s in UNIV:
 p='../persistent/stock_data/'+s+'.csv'
 try:
  d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float)
  px[s]=d[d.index<=end]
 except: pass
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# interpretable acceleration: recent 20d return relative to prior 40d return, volatility scaled
recent=P.pct_change(20); prior=P.shift(20).pct_change(40)
vol=r.rolling(60,min_periods=30).std()
f=(recent-prior)/(vol*np.sqrt(20)+1e-12)
# no lookahead: signal at t, forward return t+1..t+10
f=f.shift(1); fr=P.shift(-10)/P-1
ics=[]; dates=[]; counts=[]; turnovers=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); counts.append(len(z))
# turnover based rank direction/sign changes, cross-sectional rank turnover
for dt0,dt1 in zip(dates[:-1],dates[1:]):
 a=f.loc[dt0].rank(pct=True); b=f.loc[dt1].rank(pct=True)
 z=pd.concat([a,b],axis=1).dropna()
 if len(z): turnovers.append(np.mean(abs(z.iloc[:,1]-z.iloc[:,0])))
a=np.array(ics); mean=a.mean(); sd=a.std(ddof=1)
print('factor=vol_scaled_momentum_acceleration; dates=%d instruments=%d avg_valid=%.2f period=%s..%s'%(len(a),len(P.columns),np.mean(counts),dates[0],dates[-1]))
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f coverage %.4f'%(mean,mean/sd*np.sqrt(len(a)),np.mean(a>0),np.mean(turnovers),np.mean(counts)/len(P.columns)))
for n in [120,260,520,780]:
 q=a[-n:] if len(a)>=n else a
 print('recent%d IC %.6f ICIR %.6f hit %.4f n=%d'%(n,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),np.mean(q>0),len(q)))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(aa),len(aa))
# artifact sampled all dates
out=f.copy(); out.index.name='date'; out.to_csv('scripts/artifacts/miner_1_20350329_acceleration_momentum_signal.csv')
