"""One candidate standalone validation: equity-crypto correlation-state beta contraction."""
import numpy as np,pandas as pd,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2033-06-08')
def ld(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:END]
p=pd.concat({a:ld(a) for a in A},axis=1).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
def bet(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A},index=p.index)
b60=bet(r,m,60,40); e=r-b60.mul(m,axis=0)
c=r.SPX.rolling(20,min_periods=15).corr(r.BTC); d=c.diff(); d=((d-d.rolling(60,min_periods=42).mean())/(d.rolling(60,min_periods=42).std()+1e-12)).clip(-6,6)
f=bet(e,d,60,42)-bet(e,d,20,14)
print('FACTOR equity_crypto_correlation_state_loading_contraction_60_20d end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'assets',len(A),'driver_coverage',round(d.notna().mean(),4),'factor_cells',int(f.notna().sum().sum()))
ics={};met={}
for h in (1,5,10,20):
 out=[];nn=[];fw=p.shift(-h)/p-1
 for t in p.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: out.append((t,z.f.corr(z.y,method='spearman')));nn.append(len(z))
 x=pd.Series(dict(out),dtype=float); ics[h]=x; sd=x.std(); met[h]=(x.mean(),x.mean()/sd,sd,(x>0).mean(),len(x),np.mean(nn))
 print('H',h,'IC %.6f ICIR %.6f SD %.6f HIT %.4f DATES %d MEAN_N %.2f'%met[h])
for name,lo,hi in [('2020_24','2020','2025'),('2025_26','2025','2027'),('2027_onward','2027','2100')]:
 x=ics[10][(ics[10].index>=lo)&(ics[10].index<hi)];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(np.mean(to),6),'TURN_DATES',len(to),'DECAY',json.dumps({str(h):{'ic':round(met[h][0],6),'icir':round(met[h][1],6),'dates':met[h][4]}for h in met}))
print('LIBRARY_CORRELATION: NOT COMPUTED; standalone candidate cannot be admitted without current-library evidence')
