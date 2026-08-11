import pandas as pd, numpy as np, glob, os

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
macro = ['DXY','USDCNY','USDJPY','EURUSD','VIX']
rows=[]
for a in assets:
    p = f'../persistent/stock_data/{a}.csv'
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'] <= '2026-07-29']
    c = df['close'].astype(float)
    last = c.iloc[-1]
    def chg(n):
        return (last/c.iloc[-1-n]-1)*100 if len(c)>n else np.nan
    r = c.pct_change()
    vol20 = r.iloc[-20:].std()*np.sqrt(252)*100
    ma20 = c.iloc[-20:].mean(); ma60 = c.iloc[-60:].mean() if len(c)>=60 else np.nan
    rows.append(dict(asset=a, last=round(last,2), r5=round(chg(5),2), r10=round(chg(10),2),
                     r20=round(chg(20),2), r30=round(chg(30),2), vol20=round(vol20,1),
                     ma20_ratio=round(last/ma20,3), ma60_ratio=round(last/ma60,3) if not np.isnan(ma60) else np.nan,
                     pos_ma20=1 if last>ma20 else 0, pos_ma60=1 if (not np.isnan(ma60) and last>ma60) else 0))
for a in macro:
    p = f'../persistent/index_data/{a}.csv'
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'] <= '2026-07-29']
    c = df['close'].astype(float)
    last = c.iloc[-1]
    def chg(n):
        return (last/c.iloc[-1-n]-1)*100 if len(c)>n else np.nan
    r = c.pct_change()
    vol20 = r.iloc[-20:].std()*np.sqrt(252)*100
    rows.append(dict(asset=a+'(obs)', last=round(last,2), r5=round(chg(5),2), r10=round(chg(10),2),
                     r20=round(chg(20),2), r30=round(chg(30),2), vol20=round(vol20,1),
                     ma20_ratio=round(last/c.iloc[-20:].mean(),3), ma60_ratio=round(last/c.iloc[-60:].mean(),3) if len(c)>=60 else np.nan,
                     pos_ma20=1 if last>c.iloc[-20:].mean() else 0, pos_ma60=1 if (len(c)>=60 and last>c.iloc[-60:].mean()) else 0))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print()
# Cross-asset dispersion & correlation regime (avg pairwise corr of 10d returns, last 60d vs prior 60d)
closes={}
for a in assets:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date']=pd.to_datetime(df['date']); df=df.sort_values('date')
    closes[a]=df.set_index('date')['close'].astype(float)
px = pd.DataFrame(closes).loc[:'2026-07-29']
r10 = px.pct_change(10).dropna()
recent = r10.iloc[-60:]; prior = r10.iloc[-120:-60]
def avg_corr(d):
    return d.corr().values[np.triu_indices(len(d.columns),1)].mean()
print('avg pairwise corr (10d rets) last 60d:', round(avg_corr(recent),3), '| prior 60d:', round(avg_corr(prior),3))
# momentum dispersion: cross-sectional std of 20d returns
cs_std = px.pct_change(20).iloc[-1].std()
print('cross-sectional std of 20d rets (latest):', round(cs_std*100,2),'%')
print('cross-sectional std of 20d rets (60d avg):', round(px.pct_change(20).iloc[-60:].std(axis=1).mean()*100,2),'%')
# VIX percentile over sample
vix = pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date'])
vix=vix[vix['date']<='2026-07-29'].sort_values('date')
print('VIX last:', round(vix['close'].iloc[-1],2), '| 60d ago:', round(vix['close'].iloc[-61],2), '| 90th pct:', round(vix['close'].quantile(0.9),2), '| median:', round(vix['close'].median(),2))
