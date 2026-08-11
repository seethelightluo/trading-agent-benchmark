"""miner_1 cycle 4: batch E — more low-corr candidates + regime checks for range_pos_60."""
import json
import base64
import zlib
import io
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
SIGNAL_END = pd.Timestamp("2026-07-29")
DATA_END = pd.Timestamp("2026-07-30")
MIN_ASSETS_PER_DATE = 8

closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
for a in ASSETS:
    df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= DATA_END].set_index("date").sort_index()
    closes[a] = df["close"].astype(float)
    vols[a] = df["volume"].astype(float)
    opens[a] = df["open"].astype(float)
    highs[a] = df["high"].astype(float)
    lows[a] = df["low"].astype(float)
close = pd.DataFrame(closes)

def f_beta_asset(asset_ret, macro_ret, win=60, minp=None):
    common = asset_ret.index.intersection(macro_ret.index)
    r_a = asset_ret.reindex(common).dropna()
    r_m = macro_ret.reindex(common).dropna()
    idx = r_a.index.intersection(r_m.index)
    r_a, r_m = r_a.reindex(idx), r_m.reindex(idx)
    minp = minp or max(30, win // 2)
    cov = r_a.rolling(win, min_periods=minp).cov(r_m)
    var = r_m.rolling(win, min_periods=minp).var()
    return cov / var

def make_beta_factor(macro_close, win=60, minp=None):
    out = {}
    for a in ASSETS:
        idx = closes[a].dropna().index
        c = closes[a].reindex(idx)
        asset_ret = c.pct_change()
        m_aligned = macro_close.reindex(idx).ffill()
        macro_ret = m_aligned.pct_change()
        out[a] = f_beta_asset(asset_ret, macro_ret, win, minp)
    return pd.DataFrame({a: s.reindex(close.index) for a, s in out.items()})

def fwd_returns(horizon):
    out = {}
    for a in ASSETS:
        c = closes[a].dropna()
        out[a] = (c.shift(-horizon) / c - 1.0).reindex(close.index)
    return pd.DataFrame(out)

def ic_series(factor, fwd_ret, min_assets=MIN_ASSETS_PER_DATE):
    dates, ics = [], []
    for dt in factor.index:
        x, y = factor.loc[dt], fwd_ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= min_assets:
            ics.append(x[m].rank().corr(y[m].rank()))
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def turnover_rank(factor, lag=10):
    return float(factor.rank(axis=1).diff(lag).abs().mean(axis=1).dropna().mean())

def load_active_library():
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        d = json.load(open(f"factors/{fid}.json"))
        raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
        p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
        lib[fid] = p
    return lib

LIB = load_active_library()

def lib_corr_breakdown(panel):
    out = {}
    for fid, lp in LIB.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        out[fid] = round(float(np.corrcoef(a[m], b[m])[0, 1]), 4) if m.sum() > 200 else None
    return out

def validate(panel, horizon=10, regimes=True):
    panel = panel[panel.index <= SIGNAL_END]
    fr = fwd_returns(horizon)
    ic = ic_series(panel, fr)
    ic_mean = float(ic.mean())
    icir = float(ic.mean() / ic.std()) if len(ic) > 2 else np.nan
    hit = float((ic > 0).mean()) if np.isfinite(ic_mean) else np.nan
    if ic_mean < 0:
        hit = float((ic < 0).mean())
    out = {"ic": round(ic_mean, 4), "icir": round(icir, 4), "hit": round(hit, 4), "n": int(len(ic)),
           "turnover": round(turnover_rank(panel), 2), "libcorr": lib_corr_breakdown(panel)}
    if regimes:
        half = len(ic) // 2
        out["ic_h1"] = round(float(ic.iloc[:half].mean()), 4)
        out["ic_h2"] = round(float(ic.iloc[half:].mean()), 4)
        m2024 = ic[ic.index >= "2024-01-01"]
        out["ic_2024plus"] = round(float(m2024.mean()), 4) if len(m2024) > 20 else None
        m2025 = ic[ic.index >= "2025-01-01"]
        out["ic_2025plus"] = round(float(m2025.mean()), 4) if len(m2025) > 20 else None
    return out

def run_dense(fn, **kw):
    out = {}
    for a in ASSETS:
        idx = closes[a].dropna().index
        try:
            s = fn(closes[a].reindex(idx), vols[a].reindex(idx), opens[a].reindex(idx),
                   highs[a].reindex(idx), lows[a].reindex(idx), **kw)
            out[a] = s
        except Exception:
            out[a] = pd.Series(np.nan, index=idx)
    return pd.DataFrame({a: s.reindex(close.index) for a, s in out.items()})

# ---- batch E candidates
def f_aroon(close_, vol_, open_, high_, low_, win=25):
    hh = high_.rolling(win).apply(lambda x: np.argmax(x), raw=True)
    ll = low_.rolling(win).apply(lambda x: np.argmin(x), raw=True)
    up = (win - hh) / win * 100.0
    dn = (win - ll) / win * 100.0
    return up - dn

def f_variance_ratio(close_, vol_, open_, high_, low_, win=60, lag=5):
    r = close_.pct_change()
    vr = r.rolling(win).var() * lag / (r.rolling(win).sum().rolling(win).var().replace(0, np.nan))
    return np.log(vr)  # >0 trending, <0 mean-reverting

def f_dispersion(close_, vol_, open_, high_, low_, win=60):
    # cross-sectional: |asset return - mean return| accumulated over win
    cs_ret = close.pct_change()
    dev = (cs_ret.sub(cs_ret.mean(axis=1), axis=0)).abs()
    return dev[a].rolling(win).sum()

def f_bb_width(close_, vol_, open_, high_, low_, win=20, long=60):
    mid = close_.rolling(win).mean()
    sd = close_.rolling(win).std()
    width = (2 * sd) / mid
    return width / width.rolling(long).mean()

def f_vol_zscore(close_, vol_, open_, high_, low_, short=20, long=60):
    return vol_.rolling(short).mean() / vol_.rolling(long).mean() - 1.0

# ---- run
CAND = {}
CAND["sox_beta_60"] = make_beta_factor(closes["SOX"], 60)
CAND["copper_beta_60"] = make_beta_factor(closes["COPPER"], 60)
CAND["cn10y_beta_60"] = make_beta_factor(closes["CN10Y"], 60)
comm = pd.concat([closes["WTI"], closes["XAU"], closes["COPPER"]]).groupby(level=0).mean().sort_index()
CAND["commodity_avg_beta_60"] = make_beta_factor(comm, 60)
CAND["aroon_25"] = run_dense(f_aroon, win=25)
CAND["variance_ratio_60"] = run_dense(f_variance_ratio, win=60)
CAND["bb_width_20x60"] = run_dense(f_bb_width)

# dispersion needs cross-sectional dev
cs_ret = close.pct_change()
dev = cs_ret.sub(cs_ret.mean(axis=1), axis=0).abs()
disp = {}
for a in ASSETS:
    idx = closes[a].dropna().index
    disp[a] = dev[a].reindex(idx).rolling(60).sum()
CAND["dispersion_60"] = pd.DataFrame({a: s.reindex(close.index) for a, s in disp.items()})

for name, panel in CAND.items():
    r = validate(panel)
    ok = abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084
    print(f"{name}: ic={r['ic']} icir={r['icir']} hit={r['hit']} n={r['n']} "
          f"turn={r['turnover']} h1={r['ic_h1']} h2={r['ic_h2']} 2024+={r['ic_2024plus']} "
          f"2025+={r['ic_2025plus']} libcorr={r['libcorr']} -> {'PASS' if ok else 'FAIL'}")

print("\n=== range_pos_60 regime check ===")
def f_range_pos(close_, vol_, open_, high_, low_, win=60):
    hh = high_.rolling(win).max()
    ll = low_.rolling(win).min()
    return (close_ - ll) / (hh - ll)
rp = run_dense(f_range_pos, win=60)
print(validate(rp))

print("\n=== btc_beta_90 & 60 regime check (confirmation) ===")
for name, win in [("btc_beta_60", 60), ("btc_beta_90", 90)]:
    p = make_beta_factor(closes["BTC"], win)
    r = validate(p)
    print(f"{name}: ic={r['ic']} icir={r['icir']} hit={r['hit']} n={r['n']} "
          f"h1={r['ic_h1']} h2={r['ic_h2']} 2024+={r['ic_2024plus']} 2025+={r['ic_2025plus']}")
