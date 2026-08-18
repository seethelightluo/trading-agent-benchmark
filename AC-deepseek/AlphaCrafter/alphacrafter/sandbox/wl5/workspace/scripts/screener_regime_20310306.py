"""Screener regime assessment - date-gated through 2031-03-05 (visible_through).
Pure data analysis; no backtest/step imports, no account/date mutation.
"""
import pandas as pd, numpy as np, glob, os

AS_OF = "2031-03-05"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load(fp):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    dcol = "date" if "date" in df.columns else df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol].astype(str).str[:10])
    df = df.sort_values(dcol).set_index(dcol)
    df = df[~df.index.duplicated(keep="last")]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

rows = []
closes = {}
for a in ASSETS:
    fp = f"../persistent/stock_data/{a}.csv"
    if not os.path.exists(fp):
        print("MISSING", fp); continue
    df = load(fp)
    df = df[df.index <= AS_OF]
    c = df["close"]
    closes[a] = c
    px = c.iloc[-1]
    r10 = c.iloc[-1]/c.iloc[-11]-1 if len(c)>11 else np.nan
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c)>21 else np.nan
    r60 = c.iloc[-1]/c.iloc[-61]-1 if len(c)>61 else np.nan
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    ret = c.pct_change()
    vol20 = ret.rolling(20).std().iloc[-1]*np.sqrt(252)
    vol60 = ret.rolling(60).std().iloc[-1]*np.sqrt(252)
    # vol of vol (20d std of 20d realized vol)
    rv20 = ret.rolling(20).std()*np.sqrt(252)
    vov = rv20.rolling(60).std().iloc[-1]
    rows.append(dict(asset=a, px=round(px,4), r10=round(r10*100,2), r20=round(r20*100,2), r60=round(r60*100,2),
                     vs_ma20=round((px/ma20-1)*100,2), vs_ma60=round((px/ma60-1)*100,2),
                     vol20=round(vol20*100,1), vol60=round(vol60*100,1), vov=round(vov*100,2)))

res = pd.DataFrame(rows)
print("=== TRADABLE UNIVERSE (through", AS_OF, ") ===")
print(res.to_string(index=False))

# macro
print("\n=== MACRO OBSERVATION SIGNALS ===")
for m in MACRO:
    fp = f"../persistent/index_data/{m}.csv"
    df = load(fp)
    df = df[df.index <= AS_OF]
    c = df["close"] if "close" in df.columns else df.iloc[:,1]
    px = c.iloc[-1]
    r10 = c.iloc[-1]/c.iloc[-11]-1 if len(c)>11 else np.nan
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c)>21 else np.nan
    r60 = c.iloc[-1]/c.iloc[-61]-1 if len(c)>61 else np.nan
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    print(f"{m:8s} px={px:10.4f} r10={r10*100:7.2f}% r20={r20*100:7.2f}% r60={r60*100:7.2f}% vsMA20={(px/ma20-1)*100:7.2f}% vsMA60={(px/ma60-1)*100:7.2f}%")

# cross-sectional dispersion
panel = pd.DataFrame(closes)
ret10 = panel.pct_change(10).iloc[-1]
print("\n=== CROSS-SECTIONAL 10d RETURN DISPERSION ===")
print("std:", round(ret10.std()*100, 2), "| max:", round(ret10.max()*100,2), ret10.idxmax(), "| min:", round(ret10.min()*100,2), ret10.idxmin())
print("mean:", round(ret10.mean()*100,2))

# block returns since last screener (2031-02-19 close -> 2031-03-05 close)
print("\n=== BLOCK PX CHANGE 2031-02-19 -> 2031-03-05 ===")
for a in ASSETS:
    c = closes[a]
    c = c[c.index <= AS_OF]
    prev = c[c.index <= "2031-02-19"]
    if len(prev)==0: continue
    b = c.iloc[-1]/prev.iloc[-1]-1
    print(f"{a:10s} {b*100:8.2f}%")

# 60d trend direction days
print("\n=== 60d UP-DAY COUNTS (last 60 sessions) ===")
for a in ASSETS:
    c = closes[a]; c = c[c.index <= AS_OF]
    r = c.pct_change().dropna().tail(60)
    ups = (r>0).sum()
    print(f"{a:10s} up {ups:2d}/60  last5 signs: {''.join('U' if x>0 else 'D' for x in r.tail(5).values)}")
