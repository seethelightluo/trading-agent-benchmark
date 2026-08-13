import pandas as pd, numpy as np, os
p = '../persistent/index_data/VIX.csv'
df = pd.read_csv(p)
df.columns = [c.strip() for c in df.columns]
dcol = df.columns[0]
df['_d'] = pd.to_datetime(df[dcol])
df = df[df['_d']<='2031-05-02'].set_index('_d').sort_index()
v = df[df.columns[1]].astype(float)
print("VIX last:", round(v.iloc[-1],2))
print("VIX 60d ago:", round(v.iloc[-61],2), " 120d ago:", round(v.iloc[-121],2), " 252d ago:", round(v.iloc[-253],2))
print("VIX 1y mean:", round(v.tail(252).mean(),2), " median:", round(v.tail(252).median(),2), " max:", round(v.tail(252).max(),2))
print("VIX pctile (1y):", round((v.tail(252) <= v.iloc[-1]).mean(),3))
# recent 10 days
print("\nVIX last 12 values:")
print(v.tail(12).round(2).to_string())
# correlation: XAU vs SPX last 60d to gauge risk-off
for pair in [('XAU','SPX'),('XAU','US10Y'),('US10Y','SPX')]:
    a,b = pair
    da = pd.read_csv(f'../persistent/stock_data/{a}.csv'); da.columns=[c.strip() for c in da.columns]
    db = pd.read_csv(f'../persistent/stock_data/{b}.csv'); db.columns=[c.strip() for c in db.columns]
    ra = da[da.columns[0]].astype(str); ca = da['close'].astype(float)
    rb = db[db.columns[0]].astype(str); cb = db['close'].astype(float)
    sa = pd.Series(ca.values, index=pd.to_datetime(ra)).sort_index()
    sb = pd.Series(cb.values, index=pd.to_datetime(rb)).sort_index()
    sa = sa[sa.index<='2031-05-02'].pct_change().tail(60)
    sb = sb[sb.index<='2031-05-02'].pct_change().tail(60)
    j = pd.concat([sa,sb],axis=1).dropna()
    print(f"corr({a},{b}) 60d:", round(j.corr().iloc[0,1],3), "n=", len(j))
