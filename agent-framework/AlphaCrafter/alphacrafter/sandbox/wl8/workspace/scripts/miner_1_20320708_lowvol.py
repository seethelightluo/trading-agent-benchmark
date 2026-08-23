import pandas as pd,numpy as np,json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-07-08')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change();
# lagged trailing realized volatility, smoothed to reduce one-day noise; lower volatility ranks higher.
f=-(r.rolling(60,min_periods=60).std().shift(1)).rolling(5,min_periods=5).mean(); fr=p.shift(-10)/p-1
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-05-01') or d>cut:continue
 q=ic(f.iloc[i],fr.iloc[i]);
 if pd.notna(q):rows.append((d,q,int((f.iloc[i].notna()&fr.iloc[i].notna()).sum())))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic; m=float(z.mean()); ir=float(m/z.std(ddof=1)); turn=float(f.rank(pct=True).diff().abs().mean().mean())
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'IC',m,'ICIR',ir,'hit',(z>0).mean(),'turnover',turn)
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1;q=[ic(f.loc[d],ff.loc[d]) for d in x.index];q=[v for v in q if pd.notna(v)];print('decay',h,float(np.mean(q)))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,float(q.mean()),float(q.mean()/q.std(ddof=1)),len(q))
f.loc[x.index].to_csv('scripts/miner_1_20320708_lowvol_signal.csv');x.to_csv('scripts/miner_1_20320708_lowvol_ic.csv')
print('METRICS',json.dumps({'ic':m,'icir':ir,'turnover':turn,'coverage':float(f.loc[x.index].notna().mean().mean()),'dates':len(z),'avg_n':float(x.n.mean()),'period':[str(x.index.min().date()),str(x.index.max().date())]}))
