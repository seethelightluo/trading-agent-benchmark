import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).loc[:'2033-04-13']
r=px.pct_change(); vol=r.rolling(20).std(); f0=-px.pct_change(5)/vol
# cross-sectional rank average of fast reversal and slower inverse vol momentum
rank=lambda x:x.rank(axis=1,pct=True)
vs={'invvol_rev5':rank(f0),'invvol_rev5_20':.7*rank(f0)+.3*rank(-px.pct_change(20)/px.pct_change().rolling(40).std()),'invvol_rev10':rank(-px.pct_change(10)/vol)}
def ic(a,b):
 n=(a.notna()&b.notna()).sum(axis=1); ar=a.rank(axis=1);br=b.rank(axis=1); am=ar.mean(axis=1);bm=br.mean(axis=1); cov=((ar-am.values[:,None])*(br-bm.values[:,None])).sum(axis=1); s1=np.sqrt(((ar-am.values[:,None])**2).sum(axis=1));s2=np.sqrt(((br-bm.values[:,None])**2).sum(axis=1));return (cov/s1/s2).where(n>=8).dropna()
for k,f in vs.items():
 q=ic(f,px.shift(-10)/px-1);print(k,len(q),q.mean(),q.mean()/q.std(),(q>0).mean(),[(h,round(ic(f,px.shift(-h)/px-1).mean(),6)) for h in [1,5,10,20]])
