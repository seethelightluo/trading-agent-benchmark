"""Screener cycle 2031-07-18: regime assessment + recent IC for factor ensemble refresh.
Data visible through 2031-07-17 (previous completed trading day before decision).
"""
import json, zlib, base64, io, csv, glob, os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

CURRENT = "2031-07-17"
ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

px = {}
for a in ASSETS:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CURRENT].set_index("date")["close"]
    px[a] = df
px = pd.DataFrame(px).sort_index()

mpx = {}
for s in MACRO:
    df = pd.read_csv(f"../persistent/index_data/{s}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CURRENT].set_index("date")["close"]
    mpx[s] = df
mpx = pd.DataFrame(mpx).sort_index()

print("=== price panel ===")
print("rows:", len(px), "first:", px.index.min().date(), "last:", px.index.max().date())
print("macro last:", mpx.index.max().date())

ret = px.pct_change()
last = px.iloc[-1]

# --- regime metrics ---
r5 = (px.iloc[-1] / px.iloc[-6] - 1) * 100
r20 = (px.iloc[-1] / px.iloc[-21] - 1) * 100
r60 = (px.iloc[-1] / px.iloc[-61] - 1) * 100
ma20 = px.rolling(20).mean().iloc[-1]
ma60 = px.rolling(60).mean().iloc[-1]
ma200 = px.rolling(200).mean().iloc[-1] if len(px) > 200 else pd.Series(np.nan, index=px.columns)
vol20 = ret.tail(20).std() * np.sqrt(252) * 100
vol60 = ret.tail(60).std() * np.sqrt(252) * 100
mean20 = ret.tail(20).mean() * 100

breadth = (last > ma20).mean()
above_ma60 = (last > ma60).mean()
above_ma200 = (last > ma200).mean()

print("\n=== regime @", CURRENT, "===")
print(f"20d mean daily ret: {mean20.mean():.4f}%  (asset avg)")
print(f"breadth > MA20: {breadth:.1%}  > MA60: {above_ma60:.1%}  > MA200: {above_ma200:.1%}")
print(f"avg 20d ann vol: {vol20.mean():.1f}%  median: {vol20.median():.1f}%  | 60d vol: {vol60.mean():.1f}%")
print(f"avg 5d/20d/60d ret: {r5.mean():.2f}% / {r20.mean():.2f}% / {r60.mean():.2f}%")
print("\nasset 20d ret % / 20d vol % / above MA20 / above MA200:")
for a in ASSETS:
    print(f"  {a:10s} r20={r20[a]:7.2f}  vol20={vol20[a]:5.1f}  >MA20={'Y' if last[a]>ma20[a] else 'N'}  >MA200={'Y' if last[a]>ma200[a] else 'N'}")

# macro
print("\n=== macro ===")
mlast = mpx.iloc[-1]
for s in MACRO:
    m5 = (mpx[s].iloc[-1] / mpx[s].iloc[-6] - 1) * 100 if len(mpx[s]) > 6 else np.nan
    m20 = (mpx[s].iloc[-1] / mpx[s].iloc[-21] - 1) * 100 if len(mpx[s]) > 21 else np.nan
    print(f"  {s:8s} last={mlast[s]:9.2f}  5d={m5:7.2f}%  20d={m20:7.2f}%")

# --- recent IC ---
def fwd_ret(h):
    return px.shift(-h) / px - 1.0
fwd1, fwd5, fwd10 = fwd_ret(1), fwd_ret(5), fwd_ret(10)

def load_factor_signal(path):
    d = json.load(open(path))
    sa = d.get("validation", {}).get("signal_artifact")
    if sa is None:
        sa = d.get("signal_artifact")
    if sa is None:
        return None
    raw = base64.b64decode(sa["data"])
    txt = zlib.decompress(raw).decode()
    rows = list(csv.reader(io.StringIO(txt)))
    cols = rows[0]
    sig = pd.DataFrame(rows[1:], columns=cols)
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.set_index("date")
    for c in ASSETS:
        sig[c] = pd.to_numeric(sig[c], errors="coerce")
    sig = sig[~sig.index.duplicated(keep="last")]
    return sig[ASSETS]

def daily_ic(sig, fwd):
    dates = sig.index.intersection(fwd.index)
    ics = []
    for dt in dates:
        s = sig.loc[dt]; f = fwd.loc[dt]
        mask = s.notna() & f.notna()
        if mask.sum() >= 8:
            ics.append((dt, spearmanr(s[mask], f[mask]).statistic))
    if not ics:
        return None
    return pd.Series({dt: ic for dt, ic in ics}).sort_index()

print("\n=== recent IC (through", CURRENT, ") ===")
sel = ["miner2_20260715_nclv_1d","miner2_20260715_rev_2d","miner2_20260715_rev_1d","miner2_20260715_rev_5d",
       "miner2_20260715_nclv_5d","mom_120d_skip5","vol_of_vol20x60","vix_beta_cond_60x20","miner2_20260715_nbody_1d",
       "miner2_20260715_id_rev_1d"]
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f:
        continue
    fid = os.path.basename(f)[:-5]
    if fid not in sel:
        continue
    sig = load_factor_signal(f)
    if sig is None:
        print(f"{fid:32s} NO SIGNAL")
        continue
    row = {}
    for h, fwd in [("ic1", fwd1), ("ic5", fwd5), ("ic10", fwd10)]:
        ics = daily_ic(sig, fwd)
        if ics is None:
            row[f"{h}_60"] = np.nan; row[f"{h}_20"] = np.nan; row[f"{h}ir_60"] = np.nan; row[f"{h}ir_20"] = np.nan
            continue
        s60 = ics.tail(60); s20 = ics.tail(20)
        row[f"{h}_60"] = s60.mean(); row[f"{h}_20"] = s20.mean()
        row[f"{h}ir_60"] = s60.mean() / s60.std() if s60.std() > 0 else 0.0
        row[f"{h}ir_20"] = s20.mean() / s20.std() if s20.std() > 0 else 0.0
    print(f"{fid:32s} ic1_60={row['ic1_60']: .4f} ic1_20={row['ic1_20']: .4f} | ic5_60={row['ic5_60']: .4f} ic5_20={row['ic5_20']: .4f} | ic10_60={row['ic10_60']: .4f} ic10_20={row['ic10_20']: .4f} | ic1ir_60={row['ic1ir_60']: .3f} ic10ir_60={row['ic10ir_60']: .3f}")

# save for reference
out = {"current": CURRENT, "regime": {
    "mean20": float(mean20.mean()), "breadth_ma20": float(breadth), "breadth_ma60": float(above_ma60),
    "breadth_ma200": float(above_ma200), "vol20_avg": float(vol20.mean()), "vol20_med": float(vol20.median()),
    "r5_avg": float(r5.mean()), "r20_avg": float(r20.mean()), "r60_avg": float(r60.mean())}}
json.dump(out, open("screener_regime_20310718.json", "w"), indent=1)
print("\nsaved screener_regime_20310718.json")
