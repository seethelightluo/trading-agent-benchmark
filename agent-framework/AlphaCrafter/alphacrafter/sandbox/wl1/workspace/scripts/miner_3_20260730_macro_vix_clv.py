import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
end=pd.Timestamp('2026-07-15'); assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym):
 p='../persistent/stock_data/'+sym+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+sym+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
v=load('VIX'); rows=[]
for s in assets:
 d=load(s).join(v[['close']],how='inner',rsuffix='_vix').loc[:end]
 clv=(2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan)
 z=(d.close_vix-d.close_vix.rolling(60,min_periods=30).mean())/d.close_vix.rolling(60,min_periods=30).std()
 f=clv*(1+.5*z.clip(-2,2)); r=d.close.shift(-1)/d.close-1
 rows += [(dt,s,a,b) for dt,a,b in zip(d.index,f,r) if pd.notna(a) and pd.notna(b)]
x=pd.DataFrame(rows,columns=['date','sym','f','r']); ics=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: ics.append(spearmanr(g.f,g.r).statistic)
ic=np.array(ics); counts=x.groupby('date').size(); counts=counts[counts>=8]
print('factor macro_vix_clv | dates',len(ic),'avg_n',counts.mean(),'coverage',len(x)/(len(assets)*len(counts)),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0))
for h in [5,10]:
 rr=[]
 for s in assets:
  d=load(s).join(v[['close']],how='inner',rsuffix='_vix').loc[:end]; clv=(2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan); z=(d.close_vix-d.close_vix.rolling(60,min_periods=30).mean())/d.close_vix.rolling(60,min_periods=30).std(); f=clv*(1+.5*z.clip(-2,2)); r=d.close.shift(-h)/d.close-1
  q=pd.DataFrame({'date':d.index,'f':f.values,'r':r.values}).dropna()
  for dt,g in q.groupby('date'):
   pass
  # cross-sectional by date requires collect all symbols
  
 rows_h=[]
 for s2 in assets:
  d2=load(s2).join(v[['close']],how='inner',rsuffix='_vix').loc[:end]; c=(2*d2.close-d2.high-d2.low)/(d2.high-d2.low).replace(0,np.nan); zz=(d2.close_vix-d2.close_vix.rolling(60,min_periods=30).mean())/d2.close_vix.rolling(60,min_periods=30).std(); ff=c*(1+.5*zz.clip(-2,2)); rr2=d2.close.shift(-h)/d2.close-1; rows_h += [(dt,a,b) for dt,a,b in zip(d2.index,ff,rr2) if pd.notna(a) and pd.notna(b)]
 q=pd.DataFrame(rows_h,columns=['date','f','r']); vals=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: vals.append(spearmanr(g.f,g.r).statistic)
 print('h',h,'IC',np.mean(vals),'ICIR',np.mean(vals)/np.std(vals,ddof=1),'n',len(vals))
