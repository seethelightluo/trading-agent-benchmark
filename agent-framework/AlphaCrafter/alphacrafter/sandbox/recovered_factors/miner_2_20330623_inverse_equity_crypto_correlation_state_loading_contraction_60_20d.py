"""Single pre-specified candidate: inverse SPX-BTC correlation-state residual-beta contraction."""
import numpy as np,pandas as pd,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2033-06-22')
def ld(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:END]
p=pd.concat({a:ld(a) for a in A},axis=1).sort_index().ffill();r=p.pct_change();m=r.mean(axis=1)
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A})
e=r-beta(r,m,60,40).mul(m,axis=0); c=r.SPX.rolling(20,15).corr(r.BTC); q=c.diff(); d=((q-q.rolling(60,42).mean())/(q.rolling(60,42).std()+1e-12)).clip(-6,6)
f=-(beta(e,d,60,42)-beta(e,d,20,14)); f.to_pickle('scripts/miner_2_20330623_inverse_equity_crypto_corr_state_signal.pkl')
print('FACTOR inverse_equity_crypto_correlation_state_loading_contraction_60_20d','END',END.date(),'PANEL',p.index.min().date(),p.index.max().date(),'ASSETS',len(A),'CELLS',int(f.notna().sum().sum()),'COVERAGE',round(f.notna().mean().mean(),6),'DRIVER_COVERAGE',round(d.notna().mean(),6))
ics={}; mm={}
for h in [1,5,10,20]:
 z=[]; ns=[]; fw=p.shift(-h)/p-1
 for t in p.index:
  x=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(x)>=8 and x.f.nunique()>1:z.append((t,x.f.corr(x.y,method='spearman')));ns.append(len(x))
 s=pd.Series(dict(z)); ics[h]=s;mm[h]=(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(ns))
 print('H',h,'IC %.6f ICIR %.6f HIT %.4f DATES %d MEAN_N %.2f'%mm[h])
for n,lo,hi in [('2020_24','2020','2025'),('2025_26','2025','2027'),('2027_onward','2027','2100')]:
 s=ics[10][(ics[10].index>=lo)&(ics[10].index<hi)];print('REGIME10',n,'DATES',len(s),'IC %.6f ICIR %.6f HIT %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 x=rk.iloc[[i-1,i]].T.dropna()
 if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:ts.append(1-x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
print('TURNOVER %.6f TURN_DATES %d'% (np.mean(ts),len(ts)));print('DECAY',json.dumps({h:{'ic':round(mm[h][0],6),'icir':round(mm[h][1],6),'dates':mm[h][3]}for h in mm}))
