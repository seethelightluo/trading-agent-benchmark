import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: medium-horizon reversal scaled by downside risk, with a modest shock persistence filter.
# signal at t uses data through t-1; factor = -R20 / downside_vol20 * (1 + R5/R20 clipped), interpreted as
# reversal strongest when recent move is aligned with the medium move, avoiding noisy sign flips.
xs={}
for a in assets:
    p='../persistent/stock_data/'+a+'.csv'
    d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
    d=d.loc[d.index<=pd.Timestamp('2030-06-13')]
    r=d.close.pct_change()
    down=r.where(r<0,0.0)
    dv=down.rolling(20,min_periods=15).std()*np.sqrt(20)
    r20=d.close.pct_change(20); r5=d.close.pct_change(5)
    align=(1+(r5/r20.replace(0,np.nan)).clip(-1,1)).clip(.25,1.75)
    # shift one day: no same-day close leakage
    xs[a]=(-r20/dv*align).shift(1)
F=pd.DataFrame(xs); prices=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets})
prices=prices.loc[prices.index<=pd.Timestamp('2030-06-13')]
ics={h:[] for h in [1,5,10,20]}; dates=0; nobs=[]
for dt in F.index:
    if dt not in prices.index: continue
    vals=F.loc[dt]; n=vals.notna().sum();
    if n<8: continue
    dates+=1;nobs.append(n)
    for h in ics:
        fut=prices.shift(-h).loc[dt]/prices.loc[dt]-1
        z=pd.concat([vals,fut],axis=1).dropna()
        if len(z)>=8: ics[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('candidate downside_adjusted_reversal; dates',dates,'avgN',np.mean(nobs))
for h,v in ics.items():
    v=np.array(v); print('h',h,'n',len(v),'IC',np.nanmean(v),'ICIR',np.nanmean(v)/np.nanstd(v,ddof=1)*np.sqrt(len(v)),'hit',np.mean(v>0))
v=np.array(ics[10]); print('recent261 IC/ICIR',np.mean(v[-261:]),np.mean(v[-261:])/np.std(v[-261:],ddof=1)*np.sqrt(len(v[-261:])))
# rank turnover
ranks=F.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean(),'coverage',F.notna().mean().mean())
