"""
miner_2 cycle 3b: decode library artifacts correctly (incl. miner3 gzip dicts),
check rho of body_pos_1d vs the FULL effective library, and re-run the volume
factor candidates with per-symbol dense volume handling.
"""
import os, json, pickle, time, base64, gzip, zlib
import numpy as np
import pandas as pd

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
VALID_LO, VALID_HI = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-07-15")
MIN_NAMES = 8

cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"][SYMBOLS]; open_ = cache["open"][SYMBOLS]
high = cache["high"][SYMBOLS]; low = cache["low"][SYMBOLS]
vol = cache["vol"][SYMBOLS]
idx = close.index
lret = np.log(close).diff()

def pair_rho(a, b, min_names=MIN_NAMES):
    A = a.values.astype(float); B = b.values.astype(float)
    vals = []
    for i in range(len(A)):
        x, y = A[i], B[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_names:
            continue
        x, y = x[m], y[m]
        rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
        if rx.std() <= 1e-12 or ry.std() <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(rx, ry)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan

# ---------------- library loaders ----------------
def load_npy(pid):
    M = np.load(f"factors/{pid}.npy")
    return pd.DataFrame(M, index=idx, columns=SYMBOLS)

def decode_b64zlibcsv(payload):
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    txt = raw.decode("utf-8", errors="replace")
    rows = []
    for line in txt.splitlines():
        if not line.strip() or line.startswith("date,"):
            continue
        parts = line.split(",")
        rows.append((parts[0], [float(v) if v not in ("", "NA") else np.nan for v in parts[1:]]))
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows], columns=payload.get("columns", SYMBOLS))
    df.index = pd.to_datetime(df.index)
    return df

def decode_gzipb64(payload):
    raw = base64.b64decode(payload["data"])
    try:
        raw = gzip.decompress(raw)
    except Exception:
        try:
            raw = zlib.decompress(raw)
        except Exception:
            raise ValueError("cannot decompress")
    arr = np.frombuffer(raw, dtype="<f4").reshape(payload["n_dates"], payload["n_symbols"])
    # align: common trading days of the calendar panel within [date_start, date_end]
    start = pd.Timestamp(payload["date_start"]); end = pd.Timestamp(payload["date_end"])
    mask = (idx >= start) & (idx <= end) & close.notna().all(axis=1)
    common = idx[mask]
    n = min(len(common), arr.shape[0])
    dates = common[-n:]
    return pd.DataFrame(arr[-n:], index=dates, columns=payload.get("symbols", SYMBOLS))

def load_lib_factor(path):
    d = json.load(open(path))
    sa = d.get("signal_artifact")
    if sa is None:
        v = d.get("validation") or {}
        sa = v.get("signal_artifact")
        if sa is None:
            sa = (v.get("metrics") or {}).get("signal_artifact")
    if isinstance(sa, str):
        p = os.path.join("factors", sa)
        if p.endswith(".npy") and os.path.exists(p):
            M = np.load(p)
            return pd.DataFrame(M, index=idx, columns=SYMBOLS)
        return None
    if isinstance(sa, dict):
        fmt = str(sa.get("format", ""))
        if "gzip" in fmt and "n_dates" in sa:
            try:
                return decode_gzipb64(sa)
            except Exception:
                pass
        if "base64:zlib:csv" in fmt or ("data" in sa and "n_dates" not in sa):
            try:
                return decode_b64zlibcsv(sa)
            except Exception:
                pass
    return None

lib = {}
for f in sorted(os.listdir("factors")):
    if not f.endswith(".json"):
        continue
    p = os.path.join("factors", f)
    df = load_lib_factor(p)
    if df is not None and len(df) > 100:
        lib[f] = df
print("full library loaded:")
for k, v in lib.items():
    print(f"  {k:42s} shape={v.shape} dates={v.index.min().date()}..{v.index.max().date()} finite={np.isfinite(v.values).sum()}")

# ---------------- rebuild body_pos_1d ----------------
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([open_[c], close[c], high[c], low[c]], axis=1).dropna()
    if len(df) < 30:
        continue
    rng = (df.iloc[:, 2] - df.iloc[:, 3])
    _p.loc[df.index, c] = (df.iloc[:, 1] - df.iloc[:, 0]) / rng.replace(0, np.nan)
body_pos = _p

print("\nrho(body_pos_1d, library):")
for k, v in lib.items():
    r = pair_rho(body_pos, v)
    print(f"  vs {k:42s} rho={r:.3f}  {'<<< HIGH' if r >= 0.5 else 'ok'}")

# also clv_5d (my candidate) vs full library
_p5 = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([close[c], high[c], low[c]], axis=1).dropna()
    if len(df) < 30:
        continue
    h5 = df.iloc[:, 1].rolling(5).max(); l5 = df.iloc[:, 2].rolling(5).min()
    rng = (h5 - l5).replace(0, np.nan)
    _p5.loc[df.index, c] = (df.iloc[:, 0] - l5) / rng
clv5 = _p5
print("\nrho(clv_5d_candidate, library):")
for k, v in lib.items():
    r = pair_rho(clv5, v)
    print(f"  vs {k:42s} rho={r:.3f}  {'<<< HIGH' if r >= 0.5 else 'ok'}")

# ---------------- fixed volume factors ----------------
def dense_vol_z(c, win_s=5, win_l=60):
    v = vol[c].dropna()
    if len(v) < win_l + 10:
        return pd.Series(np.nan, index=vol.index)
    mu = v.rolling(win_l).mean(); sd = v.rolling(win_l).std()
    z = (v - mu) / sd
    return z.reindex(vol.index)

volz = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    volz[c] = dense_vol_z(c, 5, 60)
vol_confirmed = -lret * volz
amihud = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([lret[c], vol[c]], axis=1).dropna()
    if len(df) < 60 or df.iloc[:, 1].abs().sum() == 0:
        continue
    a = (df.iloc[:, 0].abs() / df.iloc[:, 1]).replace([np.inf, -np.inf], np.nan)
    amihud.loc[df.index, c] = -a.rolling(20).mean()

def fwd_log(closes, h):
    return np.log(closes.shift(-h)) - np.log(closes)

def fast_ic(factor_df, fwd, min_names=MIN_NAMES):
    F = factor_df.values.astype(float); R = fwd.values.astype(float)
    n = np.isfinite(F) & np.isfinite(R)
    ok = n.sum(axis=1) >= min_names
    if not ok.any():
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    Fm = np.where(n, F, 0.0); Rm = np.where(n, R, 0.0)
    cnt = n.sum(axis=1)[ok]
    sx = Fm[ok].sum(axis=1); sy = Rm[ok].sum(axis=1)
    sxx = (Fm[ok] ** 2).sum(axis=1); syy = (Rm[ok] ** 2).sum(axis=1)
    sxy = (Fm[ok] * Rm[ok]).sum(axis=1)
    with np.errstate(all="ignore"):
        num = cnt * sxy - sx * sy
        den = np.sqrt((cnt * sxx - sx * sx) * (cnt * syy - sy * sy))
        ic = num / den
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    return {"n_dates": int(len(ic)), "n_obs": int(cnt.sum()),
            "ic": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            "hit": float((ic > 0).mean())}

m = (idx >= VALID_LO) & (idx <= VALID_HI)
fwd1 = fwd_log(close, 1)
for nm, p in [("volz_5_60_fixed", volz), ("vol_confirmed_rev_fixed", vol_confirmed), ("amihud_neg_20", amihud)]:
    ic1 = fast_ic(p.loc[m], fwd1.loc[m])
    cov = float(p.loc[m].notna().sum().sum()) / (len(SYMBOLS) * int(m.sum()))
    print(f"{nm:26s} cov={cov:.3f} IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit={ic1['hit']:.3f} n={ic1['n_dates']}")
    if ic1['n_dates'] > 0 and abs(ic1['ic']) >= 0.007 and abs(ic1['icir']) >= 0.084:
        print("  rho vs library:")
        for k, v in lib.items():
            r = pair_rho(p, v)
            print(f"    vs {k:42s} rho={r:.3f}")

print("\ndone")
