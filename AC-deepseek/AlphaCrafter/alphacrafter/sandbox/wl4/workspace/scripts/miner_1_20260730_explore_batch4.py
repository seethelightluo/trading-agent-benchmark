"""miner_1 batch4 exploration 2026-07-30 (corrected per-asset-calendar methodology).

Key fix vs earlier runs: every asset has its OWN trading calendar inside a union
calendar-daily index (BTC/ETH trade every day; equities/commodities/yields only
on their own weekdays, max run ~13).  Rolling/shift/fwd-return statistics must
be computed on each asset's own calendar (dropna -> apply -> reindex), exactly
like miner_2's clean_panel/clean_rets/fwd_panel.

Admission gate (15-asset universe, h=10): |IC|>=0.0070 and |ICIR|>=0.0840.
Also report yearly IC (regime robustness) and library correlation.
"""
import sys, math
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from factor_research_lib import TRADABLE, MACRO

END = pd.Timestamp("2026-07-29")   # visible through previous completed trading day
MIN_ASSETS = 8
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)

WL = TRADABLE


def load_asset(sym):
    df = get_stock_daily_data(symbol=sym, days=3000)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END].set_index("date")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["close"].notna()]


frames = {s: load_asset(s) for s in WL}
print(f"assets loaded: {sum(1 for v in frames.values() if v is not None)}/{len(WL)}", flush=True)

mac = {}
for m in MACRO:
    d = pd.read_csv(f"../persistent/index_data/{m}.csv")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= END].set_index("date")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    mac[m] = d["close"].dropna()

idx = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values() if f is not None])))
closes = pd.DataFrame({s: frames[s]["close"].reindex(idx) for s in WL}, index=idx)
opens = pd.DataFrame({s: frames[s]["open"].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
highs = pd.DataFrame({s: frames[s]["high"].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
lows = pd.DataFrame({s: frames[s]["low"].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
volume = pd.DataFrame({s: frames[s]["volume"].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
rets = closes.pct_change()
print(f"union idx n={len(idx)} {idx[0].date()}..{idx[-1].date()}", flush=True)


def clean_panel(func, src=None):
    out = {}
    for a in WL:
        s = (src[a] if src is not None else closes[a]).dropna()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        try:
            out[a] = func(s, a).reindex(idx) if src is not None else func(s).reindex(idx)
        except Exception as e:
            out[a] = pd.Series(np.nan, index=idx)
    return pd.DataFrame(out, index=idx)


def clean_rets(func):
    out = {}
    for a in WL:
        s = closes[a].dropna().pct_change()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        try:
            out[a] = func(s).reindex(idx)
        except Exception:
            out[a] = pd.Series(np.nan, index=idx)
    return pd.DataFrame(out, index=idx)


def fwd_panel(h):
    out = {}
    for a in WL:
        s = closes[a].dropna()
        out[a] = (s.shift(-h) / s - 1.0).reindex(idx)
    return pd.DataFrame(out, index=idx)


def macro_beta(driver_close, win=60, min_obs=40, exclude_anchor=None):
    """beta of each asset's own-calendar returns on a macro driver's changes."""
    drv_ret = driver_close.pct_change()
    out = {}
    for a in WL:
        if exclude_anchor and a == exclude_anchor:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        s = closes[a].dropna().pct_change()
        d = drv_ret.reindex(s.index)
        z = pd.concat([s.rename("a"), d.rename("m")], axis=1).dropna()
        if len(z) < min_obs:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        cov = z["a"].rolling(win, min_periods=min_obs).cov(z["m"])
        var = z["m"].rolling(win, min_periods=min_obs).var()
        b = (cov / var).where(z["m"].rolling(win, min_periods=min_obs).count() >= min_obs)
        out[a] = b.reindex(idx)
    return pd.DataFrame(out, index=idx)


def rank_ic_fast(factor_panel, fwd_panel_, min_valid=8):
    fr = factor_panel.rank(axis=1)
    rr = fwd_panel_.reindex(factor_panel.index).rank(axis=1)
    fv = fr.to_numpy(dtype=float)
    rv = rr.to_numpy(dtype=float)
    dates, ics = [], []
    for i in range(len(idx)):
        m = ~(np.isnan(fv[i]) | np.isnan(rv[i]))
        if m.sum() < min_valid:
            continue
        x = fv[i][m]
        y = rv[i][m]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        ic = float(np.corrcoef(x, y)[0, 1])
        if not np.isnan(ic):
            dates.append(factor_panel.index[i])
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")


def summarize(ics):
    ic = ics.mean()
    sd = ics.std(ddof=1)
    return {"ic": round(float(ic), 4),
            "icir": round(float(ic / sd), 4) if sd > 0 else 0.0,
            "ic_hit_ratio": round(float((np.sign(ics) == 1).mean()), 3),
            "n_ic_dates": int(len(ics)),
            "ic_std": round(float(sd), 4)}


def yearly_ic(ics):
    return {str(y): round(float(g.mean()), 4) for y, g in ics.groupby(ics.index.year)}


def coverage_metrics(panel, min_valid=8):
    valid = panel.notna()
    asset_days = float(valid.sum().sum())
    total_days = float(panel.shape[0] * len(panel.columns))
    dates_ge8 = float((valid.sum(axis=1) >= min_valid).mean())
    return {"coverage_asset_days": round(asset_days / total_days, 3),
            "coverage_dates_ge8": round(dates_ge8, 3)}


def turnover_rank(panel, step=10):
    r = panel.rank(axis=1, method="average")
    d = r.diff(step).abs().mean().mean()
    return round(float(d), 3) if not math.isnan(d) else None


def max_library_corr(candidate, library):
    best, best_key = 0.0, None
    for name, lib_sig in library.items():
        both = pd.concat([candidate.stack().rename("cand"), lib_sig.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key


FWD = {h: fwd_panel(h) for h in HORIZONS}

# ---- library signals rebuilt on per-asset calendars ----
lib = {}
lib["mom_10d_skip5"] = clean_panel(lambda s: s.shift(5) / s.shift(15) - 1.0)
lib["mom_120d_skip5"] = clean_panel(lambda s: s.shift(5) / s.shift(125) - 1.0)
lib["vol_of_vol20x60"] = clean_rets(lambda r: r.rolling(20).std().rolling(60).std())
vix = mac["VIX"]
vix_ret = vix.pct_change()
vb = {}
for a in WL:
    s = closes[a].dropna().pct_change()
    d = vix_ret.reindex(s.index)
    z = pd.concat([s.rename("a"), d.rename("m")], axis=1).dropna()
    b = (z["a"].rolling(60, min_periods=40).cov(z["m"]) / z["m"].rolling(60, min_periods=40).var())
    vb[a] = b.reindex(idx)
vb_df = pd.DataFrame(vb, index=idx)
vix20 = (vix / vix.shift(20) - 1.0)
lib["vix_beta_cond_60x20"] = -vb_df * vix20.reindex(idx)
lib["rate_beta_cn10y_60d"] = macro_beta(mac.get("CN10Y") if "CN10Y" not in mac else closes["CN10Y"].dropna())
lib["eurusd_beta_60d"] = macro_beta(mac["EURUSD"])
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
lib["dn_mkt_beta_60d"] = macro_beta(dn)
print("library signals rebuilt:", list(lib.keys()), flush=True)

C = {}
# ---- distribution/shape ----
C["skew_20d"] = clean_rets(lambda r: r.rolling(20).skew())
C["kurt_20d"] = clean_rets(lambda r: r.rolling(20).kurt())
C["close_loc_20d"] = clean_panel(
    lambda s: ((s - lows[a].reindex(s.index)) / (highs[a].reindex(s.index) - lows[a].reindex(s.index)).replace(0, np.nan)).rolling(20).mean())
# ---- vol term / risk ----
C["vol_term_20_60"] = clean_rets(lambda r: r.rolling(20).std() / r.rolling(60).std())
C["rv_ratio_5_20"] = clean_rets(lambda r: r.rolling(5).std() / r.rolling(20).std())
C["maxdd_60"] = clean_panel(lambda s: s / s.rolling(60).max() - 1.0)
# ---- momentum/oscillators ----
C["atr_mom_20"] = clean_panel(
    lambda s: (s / s.shift(20) - 1.0) / (pd.concat([(highs[a].reindex(s.index) - lows[a].reindex(s.index)),
                                                     (highs[a].reindex(s.index) - s.shift()).abs(),
                                                     (lows[a].reindex(s.index) - s.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()))


def _rsi(s, win=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(win).mean()
    dn_ = (-d.clip(upper=0)).rolling(win).mean()
    rs = up / dn_.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


C["rsi_14"] = clean_panel(_rsi)
C["bb_pos_20x2"] = clean_panel(lambda s: (s - s.rolling(20).mean()) / (2 * s.rolling(20).std()))
# ---- macro betas ----
C["dxy_beta_60d"] = macro_beta(mac["DXY"])
C["usdjpy_beta_60d"] = macro_beta(mac["USDJPY"])
C["wti_beta_60d"] = macro_beta(closes["WTI"].dropna(), exclude_anchor="WTI")
sp = closes["CN10Y"] - closes["US10Y"]
C["cnus_spread_beta_60d"] = macro_beta(sp)
# ---- liquidity/volume ----
C["volume_z_20d"] = clean_panel(
    lambda s: ((volume[a].reindex(s.index) - volume[a].reindex(s.index).rolling(20).mean())
               / volume[a].reindex(s.index).rolling(20).std()))


def _amihud(r):
    v = volume[a].reindex(r.index)
    return (r.abs() / v).rolling(20).mean()


C["amihud_illiq_20"] = clean_rets(_amihud)
C["range_eff_20"] = clean_panel(
    lambda s: ((highs[a].reindex(s.index) - lows[a].reindex(s.index)) / s).rolling(20).mean()
    / s.pct_change().rolling(20).std())

print(f"{len(C)} candidates built", flush=True)

rows = []
for k, (name, panel) in enumerate(C.items()):
    panel = panel.reindex(idx)
    ics = rank_ic_fast(panel, FWD[H_ADM])
    m = summarize(ics)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = {str(h): round(float(rank_ic_fast(panel, FWD[h]).mean()), 4) for h in HORIZONS}
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    yic = yearly_ic(ics)
    m["yearly_ic"] = yic
    m["n_neg_years"] = sum(1 for y, v in yic.items() if v < 0)
    passes = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    rows.append((name, m, passes))
    print(f"[{k+1}/{len(C)}] {name:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} "
          f"covD8={m['coverage_dates_ge8']:.3f} to={m['turnover_10d_rank']:.2f} "
          f"rho={corr:.3f}({key}) negY={m['n_neg_years']} {'PASS' if passes else ''}", flush=True)

print("\n--- ranked by |ICIR| ---", flush=True)
for name, m, passes in sorted(rows, key=lambda r: abs(r[1]["icir"]), reverse=True):
    flag = "PASS" if passes else ""
    print(f"{flag:4s} {name:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"rho={m['max_abs_library_correlation']:.3f} negY={m['n_neg_years']} "
          f"decay={m['decay_ic_by_horizon']}", flush=True)
