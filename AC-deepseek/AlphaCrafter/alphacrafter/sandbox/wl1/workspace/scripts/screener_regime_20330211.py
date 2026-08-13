"""Screener regime assessment - data through 2033-02-10 (visible_through)."""
import pandas as pd, numpy as np, os

ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"
END = "2033-02-10"

def load(sym, folder=DATA):
    df = pd.read_csv(os.path.join(folder, sym + ".csv"))
    df.columns = [c.strip() for c in df.columns]
    datecol = "date" if "date" in df.columns else df.columns[0]
    df[datecol] = pd.to_datetime(df[datecol].astype(str).str[:10])
    df = df.set_index(datecol).sort_index()
    df = df[df.index <= END]
    col = "close" if "close" in df.columns else df.columns[1]
    return df[col].astype(float)

closes = {}
for a in ASSETS:
    s = load(a)
    closes[a] = s
    print(f"{a}: last={s.index[-1].date()} n={len(s)} close={s.iloc[-1]:.4f}")

px = pd.DataFrame(closes).dropna(how="all")
rets = px.pct_change().dropna(how="all")

print("\n=== CROSS-ASSET REGIME (through", END, ") ===")
r20 = px.iloc[-1] / px.iloc[-21] - 1
r60 = px.iloc[-1] / px.iloc[-61] - 1 if len(px) > 61 else np.nan
ma20 = px.rolling(20).mean().iloc[-1]
ma60 = px.rolling(60).mean().iloc[-1] if len(px) > 60 else np.nan
above20 = (px.iloc[-1] > ma20).sum()
above60 = (px.iloc[-1] > ma60).sum() if len(px) > 60 else np.nan
mean20 = rets.tail(20).mean().sum()
mean60 = rets.tail(60).mean().sum() if len(rets) >= 60 else np.nan
vol20 = rets.tail(20).std() * np.sqrt(252)
disp20 = rets.tail(20).std(axis=1).mean()
last_disp = rets.tail(20).std(axis=1).iloc[-1]

print(f"equal-weight 20d mean daily ret: {mean20*100:+.3f}% | 60d: {mean60*100:+.3f}%")
print(f"breadth above MA20: {above20}/15 | above MA60: {above60}/15")
print(f"mean 20d ann vol: {vol20.mean()*100:.1f}% | median: {vol20.median()*100:.1f}%")
print(f"20d avg daily cross-sectional dispersion: {disp20*100:.3f}% | last: {last_disp*100:.3f}%")

tab = pd.DataFrame({"r20": r20*100, "r60": r60*100, "vol20_ann": vol20*100, "above_ma20": px.iloc[-1]>ma20, "above_ma60": px.iloc[-1]>ma60 if len(px)>60 else np.nan})
print("\n=== PER-ASSET ===")
print(tab.round(2).sort_values("r20", ascending=False).to_string())

print("\n=== MACRO (observation-only) ===")
for m in ["VIX","DXY","USDCNY","USDJPY","EURUSD"]:
    try:
        s = load(m, IDX)
        s = s[s.index <= END]
        if len(s) > 61:
            print(f"{m}: last={s.iloc[-1]:.3f} 20d chg={(s.iloc[-1]/s.iloc[-21]-1)*100:+.1f}% 60d chg={(s.iloc[-1]/s.iloc[-61]-1)*100:+.1f}%")
        elif len(s) > 21:
            print(f"{m}: last={s.iloc[-1]:.3f} 20d chg={(s.iloc[-1]/s.iloc[-21]-1)*100:+.1f}%")
        else:
            print(f"{m}: last={s.iloc[-1]:.3f} n={len(s)}")
    except Exception as e:
        print(m, "ERR", e)

print("\n=== 10d rolling cross-asset mean (last 30d) ===")
rm = rets.tail(30).mean(axis=1) * 100
print(rm.round(3).to_string())
