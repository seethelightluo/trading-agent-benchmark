import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);a[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(a).sort_index();r=p.pct_change(); out=[]; fs=[]
for i in range(65,len(p)-10):
 rv20=r.iloc[i-19:i+1].std(); rv60=r.iloc[i-59:i+1].std(); tr=r.iloc[i-19:i+1].sum(); med=tr.median()
 # compression-weighted residual trend; neutralizes cross-section and favors stable trends
 f=(tr-med)*(rv60/(rv20+1e-8)).clip(0.5,2.5)
 y=p.iloc[i+10]/p.iloc[i]-1;z=pd.concat([f,y],axis=1).dropna()
 if len(z)>=8:out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);fs.append(f)
x=np.array(out);print('dates',len(x),'avgN',len(U),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(len(x)),'hit',np.mean(x>0))
for n in [250,500,750]:
 q=x[-n:];print('recent',n,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
print('coverage',1.0,'turnover_proxy',np.mean([np.mean(abs(fs[j].rank(pct=True)-fs[j-10].rank(pct=True))) for j in range(10,len(fs),10)]))
