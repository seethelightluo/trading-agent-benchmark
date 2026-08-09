import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
from pathlib import Path

files=glob.glob('../persistent/stock_data/*.csv')
frames={Path(f).stem: pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}
px=pd.DataFrame(frames).sort_index().ffill()
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=px[[a for a in assets if a in px.columns]]
r=px.pct_change()
# Candidate: relative recovery after broad selloff. On each asset, average relative return
# over the 5 sessions following market breadth shock dates during trailing 40 sessions;
# higher recovery should predict forward returns. Strictly trailing, no lookahead.
market=r.mean(axis=1)
breadth=(r<0).mean(axis=1)
shock=(market < market.rolling(60,min_periods=40).quantile(.20)) & (breadth>=.60)
# recovery observations are shock day + next 4 sessions, but factor at t only uses completed returns <=t
# score = mean asset return - median market return on shock-window sessions, normalized by asset vol
rel=r.sub(market,axis=0)
shock_rel=rel.where(shock, np.nan)
rec=shock_rel.rolling(40,min_periods=3).mean()
vol=r.rolling(20,min_periods=15).std()
factor=rec/vol
# prevent using current-day data in forward relationship: factor at t, forward t+1..t+h
out=[]
for h in [1,5,10,20]:
    f=factor
    fw=px.shift(-h)/px-1
    vals=[]; dates=[]; ns=[]
    for d in f.index:
        x=f.loc[d]; y=fw.loc[d]
        ok=x.notna()&y.notna()
        if ok.sum()>=8:
            vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(d); ns.append(ok.sum())
    z=pd.Series(vals,index=dates).dropna()
    print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
    for label,mask in [('2020-23',z.index<'2024-01-01'),('2024-27',(z.index>='2024-01-01')&(z.index<'2028-01-01')),('2028-30',(z.index>='2028-01-01')&(z.index<'2031-01-01')),('2031',z.index>='2031-01-01'),('latest120',pd.Series(False,index=z.index))]:
        zz=z[mask] if label!='latest120' else z.tail(120)
        if len(zz)>5: print(' ',label,len(zz),round(zz.mean(),6),round(zz.mean()/zz.std(ddof=1),6))
# rank turnover sampled every 10 sessions
rank=factor.rank(axis=1,pct=True)
turn=(rank.diff(10).abs().mean(axis=1)).dropna()
print('turnover10',round(turn.mean(),6),'coverage_cells',int(factor.notna().sum().sum()),'total',factor.size,'coverage',round(factor.notna().mean().mean(),6),'assets',len(px.columns),'dates',len(px))
# candidate novelty proxy against simple known signals (not admission evidence)
for name,s in {'mom20':px.pct_change(20),'invvol20':-vol,'reversal5':-px.pct_change(5)}.items():
    aa=[]
    for d in factor.index:
      x=factor.loc[d]; y=s.loc[d]; ok=x.notna()&y.notna()
      if ok.sum()>=8: aa.append(spearmanr(x[ok],y[ok]).statistic)
    print('proxy_corr',name,round(np.nanmean(aa),6),round(np.nanmedian(aa),6))
