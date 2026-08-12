"""Compute recent cross-sectional IC for all factors in the library using data visible through current date."""
import json, zlib, base64, io, csv, glob, os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

CURRENT = "2030-08-01"  # visible through

ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]

# Load price closes for all assets
px = {}
for a in ASSETS:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CURRENT].set_index("date")["close"]
    px[a] = df
px = pd.DataFrame(px).sort_index()
print("price panel:", px.shape, px.index.min().date(), "->", px.index.max().date())

# forward returns
def fwd_ret(h):
    r = px.shift(-h) / px - 1.0
    return r

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
    """cross-sectional spearman IC per date"""
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
    out = pd.Series({dt: ic for dt, ic in ics}).sort_index()
    return out

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
    # full recent 120d
    ic_series = daily_ic(sig, fwd1)
    if ic_series is not None and len(ic_series) > 0:
        sub = ic_series.tail(120)
        row["ic1_120"] = sub.mean()
        row["ic1_120_ir"] = sub.mean() / (sub.std(ddof=1) + 1e-12) if sub.std(ddof=1) > 0 else 0.0
        row["n_dates"] = len(ic_series)
    results.append(row)

res = pd.DataFrame(results).set_index("factor_id")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(res.to_string())
res.to_csv("scripts/screener_recent_ic_full_latest.csv")
print("\nsaved scripts/screener_recent_ic_full_latest.csv")
