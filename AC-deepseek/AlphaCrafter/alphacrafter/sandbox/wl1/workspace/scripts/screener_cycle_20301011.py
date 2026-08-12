"""Screener cycle 2030-10-11: regime assessment + recent factor IC through visible date."""
import json, zlib, base64, io, csv, glob, os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

CURRENT = "2030-10-10"  # visible through (date.json: current_date=2030-10-11)

ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

# ---------- prices ----------
px = {}
for a in ASSETS:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CURRENT].set_index("date")["close"]
    px[a] = df
px = pd.DataFrame(px).sort_index()
mpx = {}
for a in MACRO:
    df = pd.read_csv(f"../persistent/index_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CURRENT].set_index("date")["close"]
    mpx[a] = df
mpx = pd.DataFrame(mpx).sort_index()

print("price panel:", px.shape, px.index.min().date(), "->", px.index.max().date())
print("macro panel:", mpx.shape, mpx.index.min().date(), "->", mpx.index.max().date())

# ---------- regime ----------
ret = px.pct_change()
last = px.iloc[-1]
r5 = (px.iloc[-1] / px.iloc[-6] - 1) * 100
r20 = (px.iloc[-1] / px.iloc[-21] - 1) * 100
r60 = (px.iloc[-1] / px.iloc[-61] - 1) * 100
ma20 = px.rolling(20).mean().iloc[-1]
ma60 = px.rolling(60).mean().iloc[-1]
ma200 = px.rolling(200).mean().iloc[-1]
vol20 = ret.tail(20).std() * np.sqrt(252) * 100
vol60 = ret.tail(60).std() * np.sqrt(252) * 100
mean20 = ret.tail(20).mean() * 100
ma20_dist = (last / ma20 - 1) * 100
ma200_dist = (last / ma200 - 1) * 100

reg = pd.DataFrame({
    "r5d%": r5.round(2), "r20d%": r20.round(2), "r60d%": r60.round(2),
    "ma20_dist%": ma20_dist.round(2), "ma200_dist%": ma200_dist.round(2),
    "vol20_ann%": vol20.round(1), "vol60_ann%": vol60.round(1),
    "mean20_daily%": (mean20 * 100).round(3),
}).sort_values("r20d%")
print("\n=== REGIME (visible through", CURRENT, ") ===")
print(reg.to_string())

print("\n--- cross-asset breadth ---")
print("assets above MA20:", int((ma20_dist > 0).sum()), "/", len(ASSETS))
print("assets with r20d>0:", int((r20 > 0).sum()), "/", len(ASSETS))
print("cross-asset mean 20d daily return %:", round(float(mean20.mean()), 4))
print("cross-asset median 20d daily return %:", round(float(mean20.median()), 4))
print("avg vol20 ann%:", round(float(vol20.mean()), 1), "| median:", round(float(vol20.median()), 1))
print("dispersion of 20d ret (std):", round(float(r20.std()), 2))

# macro
mret = mpx.pct_change()
print("\n--- macro (last 20d) ---")
for c in MACRO:
    m20 = (mpx[c].iloc[-1] / mpx[c].iloc[-21] - 1) * 100 if len(mpx) > 21 else np.nan
    print(f"{c:8s} last={mpx[c].iloc[-1]:10.2f} r20d={m20:+.2f}%")

# ---------- factor IC ----------
def fwd_ret(h):
    return px.shift(-h) / px - 1.0

fwd1 = fwd_ret(1)
fwd5 = fwd_ret(5)
fwd10 = fwd_ret(10)

def load_factor_signal(path):
    d = json.load(open(path))
    sa = d.get("validation", {}).get("signal_artifact")
    if sa is None:
        sa = d.get("signal_artifact")
    if sa is None:
        return None, d
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
    return sig[ASSETS], d

def daily_ic(sig, fwd):
    dates = sig.index.intersection(fwd.index)
    ics = []
    for dt in dates:
        s = sig.loc[dt]
        f = fwd.loc[dt]
        mask = s.notna() & f.notna()
        if mask.sum() >= 8:
            ic = spearmanr(s[mask], f[mask]).statistic
            ics.append((dt, ic))
    if not ics:
        return None
    return pd.Series({dt: ic for dt, ic in ics}).sort_index()

results = []
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f:
        continue
    sig, d = load_factor_signal(f)
    if sig is None:
        print("NO SIGNAL:", os.path.basename(f))
        continue
    fid = d.get("factor_id", os.path.basename(f)[:-5])
    tags = d.get("tags", [])
    row = {"factor_id": fid, "tags": ",".join(tags) if isinstance(tags, list) else str(tags)}
    for h, fwd in [("ic1", fwd1), ("ic5", fwd5), ("ic10", fwd10)]:
        ic_series = daily_ic(sig, fwd)
        if ic_series is None:
            row[f"{h}_60"] = np.nan; row[f"{h}_20"] = np.nan
            row[f"{h}ir_60"] = np.nan; row[f"{h}ir_20"] = np.nan
            continue
        for label, win in [("60", 60), ("20", 20)]:
            sub = ic_series.tail(win)
            if len(sub) == 0:
                row[f"{h}_{label}"] = np.nan; row[f"{h}ir_{label}"] = np.nan
            else:
                row[f"{h}_{label}"] = sub.mean()
                row[f"{h}ir_{label}"] = sub.mean() / (sub.std(ddof=1) + 1e-12) if sub.std(ddof=1) > 0 else 0.0
    ic_series = daily_ic(sig, fwd1)
    if ic_series is not None and len(ic_series) > 0:
        sub = ic_series.tail(120)
        row["ic1_120"] = sub.mean()
        row["ic1_120_ir"] = sub.mean() / (sub.std(ddof=1) + 1e-12) if sub.std(ddof=1) > 0 else 0.0
        row["n_dates"] = len(ic_series)
    results.append(row)

res = pd.DataFrame(results).set_index("factor_id")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)
print("\n=== RECENT FACTOR IC (visible through", CURRENT, ") ===")
print(res.to_string())
res.to_csv("scripts/screener_recent_ic_20301011.csv")
print("\nsaved scripts/screener_recent_ic_20301011.csv")

# ---------- factor pairwise correlation (last available aligned dates) ----------
print("\n=== FACTOR SIGNAL PAIRWISE CORRELATION (last 60 aligned obs) ===")
sigs = {}
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f:
        continue
    sig, d = load_factor_signal(f)
    if sig is None:
        continue
    fid = d.get("factor_id", os.path.basename(f)[:-5])
    sigs[fid] = sig.mean(axis=1)
sc = pd.DataFrame(sigs).dropna()
sc = sc.tail(60)
corr = sc.corr()
print(corr.round(2).to_string())
corr.to_csv("scripts/screener_factor_corr_20301011.csv")
