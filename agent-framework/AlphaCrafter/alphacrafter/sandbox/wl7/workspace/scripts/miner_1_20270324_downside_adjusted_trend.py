import pandas as pd, numpy as np, os, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
    f=os.path.join(base,a+'.csv')
    d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
    px[a]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index()
prices=prices.loc[:'2027-03-23']
r=prices.pct_change()
# Risk-adjusted medium trend, penalizing downside volatility; lagged by one session.
down=r.where(r<0,0).rolling(30,min_periods=20).std()
signal=((prices/prices.shift(60)-1)/(down*np.sqrt(30))).shift(1)
fwd=prices.shift(-1)/prices-1
rows=[]; sigrows=[]
for dt in signal.index:
    x=signal.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ic=spearmanr(x[ok],y[ok]).statistic
        rows.append((dt,ic,ok.sum()))
        for a in assets:
            if ok[a]: sigrows.append((dt,a,float(x[a])))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=ic.ic.mean(); sd=ic.ic.std(ddof=1); icir=mean/sd*np.sqrt(252)
# rank turnover on adjacent valid cross sections
ranks=signal.rank(pct=True,axis=1); turnover=ranks.diff().abs().mean(axis=1).mean()
print(json.dumps({'dates':len(ic),'avg_instruments':float(ic.n.mean()),'coverage':float(signal.notna().sum(axis=1).mean()/15),'ic':float(mean),'icir':float(icir),'hit_ratio':float((ic.ic>0).mean()),'turnover':float(turnover),'regime_2020_22':float(ic.loc['2020':'2022'].ic.mean()),'regime_2023_24':float(ic.loc['2023':'2024'].ic.mean()),'regime_2025_27':float(ic.loc['2025':'2027'].ic.mean())},indent=2))
for h in [5,10,20]:
    yy=prices.shift(-h)/prices-1; vals=[]
    for dt in signal.index:
      x=signal.loc[dt]; y=yy.loc[dt]; ok=x.notna()&y.notna()
      if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
    print('decay',h,float(np.nanmean(vals)),len(vals))
os.makedirs('scripts',exist_ok=True)
pd.DataFrame(sigrows,columns=['date','asset','signal']).to_csv('scripts/miner_1_20270324_downside_adjusted_trend_signal.csv',index=False)
