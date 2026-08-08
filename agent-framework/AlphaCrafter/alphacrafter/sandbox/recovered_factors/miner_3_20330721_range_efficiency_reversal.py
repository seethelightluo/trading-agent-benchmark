import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A if os.path.exists('../persistent/stock_data/'+a+'.csv')}
P=pd.DataFrame(p).sort_index(); r=P.pct_change(); vol=r.rolling(20).std();
# Explicit reversal of range-efficiency, lagged one completed day
f=(-(P.pct_change(20)/(r.abs().rolling(20).sum()+1e-12))).shift(1)
print('period',P.index.min().date(),P.index.max().date(),'assets',len(p))
def evalf(x,name):
 out={}
 for h in [1,5,10,20]:
  y=P.shift(-h)/P-1; z=[]; ds=[]; ns=[]
  for d in x.index:
   ok=x.loc[d].notna()&y.loc[d].notna()
   if ok.sum()>=8:
    z.append(spearmanr(x.loc[d,ok],y.loc[d,ok]).statistic);ds.append(d);ns.append(ok.sum())
  z=np.array(z); out[h]=(len(z),np.nanmean(z),np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),np.mean(z>0),np.mean(ns))
  print(name,h,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f meanN %.2f'%(out[h][1],out[h][2],out[h][3],out[h][4]))
 return out
m=evalf(f,'range_reversal')
# 10-day rank turnover and coverage
print('turn10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/f.size)
# library audit: reconstructed price-only admitted comparators, pooled common-cell Spearman
lib={
'ravmom20':(P.pct_change(20)/(vol+1e-12)).shift(1),
'volnormrev5':(-(P.pct_change(5)/(r.rolling(20).std()+1e-12))).shift(1),
'realizedvol20':(-vol).shift(1),
'consistency20':((P.pct_change(20)/(vol+1e-12))*(r.gt(0).rolling(20).mean()-0.5)).shift(1),
'accel20_60':((P.pct_change(20)-P.pct_change(60))/(vol+1e-12)).shift(1),
'drawdown60':(-(P/P.rolling(60).max()-1)).shift(1),
'autocorr20':r.rolling(20).corr(r.shift(1)).shift(1)
}
mx=(0,None,0)
for n,x in lib.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 print('corr',n,round(rho,6),'cells',len(q))
 if abs(rho)>mx[0]:mx=(abs(rho),n,rho)
print('MAX_CORR',mx)
for period in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 lo,hi=map(lambda s:pd.Timestamp(s+'-01-01'),period); mask=(P.index>=lo)&(P.index<=pd.Timestamp(period[1]+'-12-31'))
 x=f.loc[mask]; y=P.shift(-10).loc[mask]/P.loc[mask]-1; z=[]
 for d in x.index:
  ok=x.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(x.loc[d,ok],y.loc[d,ok]).statistic)
 z=np.array(z);print('REGIME',period,len(z),np.nanmean(z),np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12))
