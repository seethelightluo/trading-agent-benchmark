"""Miner 2: USDJPY-shock residual response persistence, one interpretable candidate."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-07-19')
def load(path):
    return pd.read_csv(path,parse_dates=['date']).set_index('date')['close'].loc[:END]
px=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in ASSETS},axis=1).sort_index()
r=px.pct_change().replace([np.inf,-np.inf],np.nan); common=r.median(axis=1)
fx=load('../persistent/index_data/USDJPY.csv').pct_change().reindex(r.index)
# At t use only [t-60,t): shocks are FX magnitude >= its own trailing 70pct.
# Signal: mean idiosyncratic next-session response following a prior USDJPY shock.
out=pd.DataFrame(np.nan,index=r.index,columns=ASSETS)
for t in range(61,len(r)):
    ix=r.index[t-60:t]; f=fx.loc[ix]; shock=f.abs()>=f.abs().quantile(.70)
    # Response is return at session s conditional on shock at s-1; both completed before t.
    resp_days=shock.shift(1,fill_value=False)
    x=common.loc[ix]
    for a in ASSETS:
        y=r.loc[ix,a]; ok=y.notna()&x.notna()
        beta=np.cov(y[ok],x[ok],ddof=1)[0,1]/np.var(x[ok],ddof=1) if ok.sum()>=30 and np.var(x[ok])>0 else np.nan
        z=(y-beta*x)[resp_days & y.notna() & x.notna()]
        if len(z)>=8: out.iloc[t,out.columns.get_loc(a)]=z.mean()
def met(h):
    fwd=px.shift(-h)/px-1; vals=[]; ds=[]; ns=[]
    for dt in out.index:
        z=pd.concat([out.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
            if np.isfinite(q): vals.append(q);ds.append(dt);ns.append(len(z))
    x=np.array(vals); return x,pd.DatetimeIndex(ds),ns
print('FACTOR usdjpy-shock-residual-response-persistence-60obs')
print('endpoint',END.date(),'cells',int(out.notna().sum().sum()),'of',out.size,'coverage',round(out.notna().mean().mean(),6))
allm={}
for h in [1,5,10,20]:
 x,ds,ns=met(h); allm[h]=(x,ds)
 print('h',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6),'dates',len(x),'mean_n',round(np.mean(ns),3))
x,ds=allm[10]
for lab,lo,hi in [('2026-2029','2026-01-01','2029-12-31'),('2030-2032','2030-01-01','2032-12-31'),('2033-end','2033-01-01',str(END.date()))]:
 z=x[(ds>=lo)&(ds<=hi)]; print('regime10',lab,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rnk=out.rank(axis=1,pct=True); d=(rnk-rnk.shift()).abs().stack()
print('turnover',round(d.mean(),6),'comparisons',len(d),'median_iqr',round(out.quantile(.75,axis=1).sub(out.quantile(.25,axis=1)).median(),6))
out.to_csv('scripts/miner_2_20340720_usdjpy_residual_response_signal.csv')
