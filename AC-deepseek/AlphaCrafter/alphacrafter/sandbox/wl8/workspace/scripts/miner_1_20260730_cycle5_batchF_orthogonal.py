"""miner_1 cycle5 batchF: orthogonal factor families screen.
Families: calendar seasonality (dow, turn-of-month), FX-macro betas (USDJPY/EURUSD,
observation-only), trend efficiency/consistency, price location, vol dynamics, kurtosis.
Admission gate: |IC10| >= 0.0070 and |ICIR10| >= 0.0840 on 15-asset universe.
Orthogonality: max abs spearman rho vs 3 current library panels must be < 0.5 (gate
recomputes from signal artifacts); we pre-screen with the same method.
"""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    ASSETS, load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, artifact_b64, IC_GATE, ICIR_GATE, CURRENT_DATE,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
macro = {
    "DXY": load_index("DXY"), "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"), "EURUSD": load_index("EURUSD"),
    "VIX": load_index("VIX"),
}
print(f"data loaded {time.time()-t0:.1f}s | panel {close.shape}", flush=True)

# ---------------- library panels (current effective) ----------------
def load_lib_panels():
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        d = json.load(open(f"factors/{fid}.json"))
        raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
        panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
        panel.index = pd.DatetimeIndex(panel.index)
        lib[fid] = panel
    return lib

lib = load_lib_panels()
print("library panels:", {k: v.shape for k, v in lib.items()}, flush=True)


def spearman_lib_corr(panel, lib_panels):
    """Max abs spearman rho between candidate panel and each library panel on
    overlapping dates/assets (pooled values), mirroring the deterministic gate."""
    out = {}
    for fid, lp in lib_panels.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            out[fid] = np.nan
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            out[fid] = np.nan
            continue
        out[fid] = abs(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()))
    return out


# ---------------- candidate factor functions ----------------
def f_dow_eff_60(c, v, o, h, l, m, win=60):
    """Day-of-week seasonality: mean return on same weekday over trailing win days
    (calendar-effect persistence)."""
    r = c.pct_change()
    out = pd.Series(np.nan, index=c.index)
    for i in range(win + 5, len(c)):
        idx = c.index[max(0, i - win):i]
        same = r.loc[idx][r.loc[idx].index.weekday == c.index[i].weekday]
        out.iloc[i] = same.mean() if len(same) > 3 else np.nan
    return out

def f_tom_dist(c, v, o, h, l, m, win=20):
    """Turn-of-month: mean of 'days-to-month-end' sign flip; factor = -days_since_month_start
    (early-month bias). Use trailing win average of dte."""
    dte = pd.Series([(c.index[i].to_period('M').end_time.normalize() - c.index[i]).days
                     for i in range(len(c.index))], index=c.index)
    return -dte

def f_tom_effect(c, v, o, h, l, m, win=60):
    """Turn-of-month seasonality: mean return in the same 5-day bucket of the month
    over trailing win months (persistence of month-phase return)."""
    r = c.pct_change()
    phase = pd.Series([min(5, (c.index[i].day - 1) // 5 + 1) for i in range(len(c.index))],
                      index=c.index)
    out = pd.Series(np.nan, index=c.index)
    for i in range(win + 5, len(c)):
        idx = c.index[max(0, i - win):i]
        same = r.loc[idx][phase.loc[idx] == phase.iloc[i]]
        out.iloc[i] = same.mean() if len(same) > 3 else np.nan
    return out

def f_usdjpy_beta(c, v, o, h, l, m, beta_win=60, fx_win=10, sign_win=20):
    """Conditional USDJPY-beta: beta of asset 10d rets to USDJPY 10d rets over beta_win,
    signed by trailing sign_win USDJPY move (risk-on/off macro propagation)."""
    fx = m["USDJPY"].reindex(c.index).ffill()
    a_r = c.pct_change(10)
    f_r = fx.pct_change(10)
    sign = fx.pct_change(sign_win)
    out = pd.Series(np.nan, index=c.index)
    for i in range(beta_win + 12, len(c)):
        s = slice(i - beta_win, i)
        av, fv = a_r.iloc[s], f_r.iloc[s]
        mm = av.notna() & fv.notna()
        if mm.sum() >= 20 and fv[mm].std() > 1e-12:
            beta = np.polyfit(fv[mm], av[mm], 1)[0]
            out.iloc[i] = beta * np.sign(sign.iloc[i] if np.isfinite(sign.iloc[i]) else 0)
    return out

def f_eurusd_beta(c, v, o, h, l, m, beta_win=60, fx_win=10, sign_win=20):
    fx = m["EURUSD"].reindex(c.index).ffill()
    a_r = c.pct_change(10)
    f_r = fx.pct_change(10)
    sign = fx.pct_change(sign_win)
    out = pd.Series(np.nan, index=c.index)
    for i in range(beta_win + 12, len(c)):
        s = slice(i - beta_win, i)
        av, fv = a_r.iloc[s], f_r.iloc[s]
        mm = av.notna() & fv.notna()
        if mm.sum() >= 20 and fv[mm].std() > 1e-12:
            beta = np.polyfit(fv[mm], av[mm], 1)[0]
            out.iloc[i] = beta * np.sign(sign.iloc[i] if np.isfinite(sign.iloc[i]) else 0)
    return out

def f_eff_ratio_20(c, v, o, h, l, m, win=20):
    """Kaufman efficiency ratio: |net move| / sum(|ret|) over win."""
    r = c.pct_change().abs()
    net = (c / c.shift(win) - 1).abs()
    eff = net / r.rolling(win).sum()
    return eff.replace([np.inf, -np.inf], np.nan)

def f_trend_consistency_20(c, v, o, h, l, m, win=20):
    """Fraction of days whose return sign matches the trailing win return sign."""
    r = c.pct_change()
    trend = np.sign(c.diff(win))
    up = (np.sign(r) == trend).astype(float)
    out = up.rolling(win).mean()
    out[trend == 0] = np.nan
    return out

def f_hl_pos_60(c, v, o, h, l, m, win=60):
    """Price location: (close - low_win)/(high_win - low_win)."""
    hi = c.rolling(win).max()
    lo = c.rolling(win).min()
    return (c - lo) / (hi - lo).replace(0, np.nan)

def f_gain_loss_20(c, v, o, h, l, m, win=20):
    """Gain/loss ratio: mean up-day ret / mean |down-day ret| over win."""
    r = c.pct_change()
    up = r.where(r > 0)
    dn = r.where(r < 0)
    ratio = up.rolling(win).mean() / dn.rolling(win).mean().abs()
    return ratio

def f_vol_ratio_5x60(c, v, o, h, l, m, short=5, long=60):
    """Short-term vol expansion: 5d realized vol / 60d realized vol."""
    r = c.pct_change()
    vs = r.rolling(short).std()
    vl = r.rolling(long).std()
    return (vs / vl).replace([np.inf, -np.inf], np.nan)

def f_down_vol_ratio_20x60(c, v, o, h, l, m, short=20, long=60):
    """Downside semi-deviation 20d / total vol 60d."""
    r = c.pct_change()
    dn = r.where(r < 0, 0.0)
    down_sd = (dn ** 2).rolling(short).mean().apply(np.sqrt)
    tot_sd = r.rolling(long).std()
    return (down_sd / tot_sd).replace([np.inf, -np.inf], np.nan)

def f_ret_kurt_30(c, v, o, h, l, m, win=30):
    """Rolling excess kurtosis of daily returns."""
    r = c.pct_change()
    return r.rolling(win).kurt()

def f_vol_concentration_5(c, v, o, h, l, m, win=5):
    """Volume concentration: Herfindahl of last win daily volume shares."""
    if v is None:
        return pd.Series(np.nan, index=c.index)
    sh = v / v.rolling(win).sum()
    return (sh ** 2).rolling(win).sum()

def f_mom_eff_10x20(c, v, o, h, l, m, mom_win=10, eff_win=20):
    """Momentum x efficiency: trend-strength-adjusted momentum."""
    r = c.pct_change().abs()
    net = (c / c.shift(mom_win) - 1)
    eff = net.abs() / r.rolling(eff_win).sum()
    return (net * eff).replace([np.inf, -np.inf], np.nan)

def f_weekend_gap(c, v, o, h, l, m, win=40):
    """Calendar: avg weekend/overnight-gap persistence - mean of Monday gap
    (open - prev close)/prev close over trailing win weeks."""
    if o is None:
        return pd.Series(np.nan, index=c.index)
    gap = o / c.shift(1) - 1
    is_mon = (c.index.weekday == 0)
    out = pd.Series(np.nan, index=c.index)
    for i in range(win, len(c)):
        idx = c.index[:i]
        mons = gap.loc[idx][is_mon.loc[idx]]
        out.iloc[i] = mons.tail(win).mean() if len(mons) >= 5 else np.nan
    return out

CANDIDATES = [
    ("dow_eff_60", f_dow_eff_60, "day-of-week seasonality persistence 60d"),
    ("tom_dist", f_tom_dist, "turn-of-month: -days since month start"),
    ("tom_effect_60", f_tom_effect, "turn-of-month phase-return persistence"),
    ("usdjpy_beta_60x10", f_usdjpy_beta, "cond USDJPY beta 60x10 signed 20d"),
    ("eurusd_beta_60x10", f_eurusd_beta, "cond EURUSD beta 60x10 signed 20d"),
    ("eff_ratio_20", f_eff_ratio_20, "Kaufman efficiency ratio 20d"),
    ("trend_consistency_20", f_trend_consistency_20, "trend-sign consistency 20d"),
    ("hl_pos_60", f_hl_pos_60, "price location in 60d range"),
    ("gain_loss_20", f_gain_loss_20, "gain/loss ratio 20d"),
    ("vol_ratio_5x60", f_vol_ratio_5x60, "vol expansion 5/60"),
    ("down_vol_ratio_20x60", f_down_vol_ratio_20x60, "downside vol ratio 20/60"),
    ("ret_kurt_30", f_ret_kurt_30, "rolling excess kurtosis 30d"),
    ("vol_concentration_5", f_vol_concentration_5, "volume Herfindahl 5d"),
    ("mom_eff_10x20", f_mom_eff_10x20, "momentum x efficiency 10/20"),
    ("weekend_gap_40", f_weekend_gap, "Monday-gap persistence 40w"),
]

HORIZONS = (1, 2, 3, 5, 10, 20)
results = {}
for i, (name, fn, desc) in enumerate(CANDIDATES):
    t1 = time.time()
    try:
        panel = factor_panel(fn, close, vol, open_, high, low, macro)
    except Exception as e:
        print(f"[{i+1}/{len(CANDIDATES)}] {name}: PANEL ERROR {e}", flush=True)
        continue
    cov_ad, cov_ge8 = coverage(panel)
    to = turnover_rank(panel)
    decay, ic_by_h = {}, {}
    for h in HORIZONS:
        fr = fwd_returns(close, h)
        ic = ic_series(panel, fr)
        ic_by_h[h] = ic
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    icm = ic_by_h[10]
    ic = float(icm.mean()) if len(icm) else np.nan
    icir = float(icm.mean() / icm.std()) if len(icm) > 2 else np.nan
    hit = float((icm > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((icm < 0).mean())
    rho_map = spearman_lib_corr(panel, lib)
    maxrho = max((v for v in rho_map.values() if np.isfinite(v)), default=np.nan)
    gate = np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    ortho = np.isfinite(maxrho) and maxrho < 0.5
    results[name] = {
        "desc": desc, "ic": round(ic, 4), "icir": round(icir, 4),
        "ic_hit_ratio": round(hit, 4), "n_ic_dates": int(len(icm)),
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_ge8, 4),
        "turnover_10d_rank": round(to, 4),
        "decay_ic_by_horizon": {str(h): round(decay[h], 4) for h in HORIZONS},
        "lib_spearman": {k: (round(v, 3) if np.isfinite(v) else None) for k, v in rho_map.items()},
        "max_abs_library_correlation": round(maxrho, 4) if np.isfinite(maxrho) else None,
    }
    flag = "PASS" if (gate and ortho) else ("GATE-OK-HI-CORR" if gate else "fail")
    print(f"[{i+1}/{len(CANDIDATES)}] {name:22s} {desc:38s} IC10={ic:+.4f} ICIR10={icir:+.4f} "
          f"hit={hit:.3f} n={len(icm)} cov={cov_ad:.3f}/{cov_ge8:.3f} to={to:.2f} "
          f"sp={ {k: round(v,2) for k,v in rho_map.items()} } maxrho={maxrho:.3f} -> {flag} ({time.time()-t1:.1f}s)",
          flush=True)

with open("scripts/_miner1_cycle5_batchF_results.json", "w") as fp:
    json.dump(results, fp, indent=1, default=str)

print("\n===== BATCH F SUMMARY (gate |IC|>=%.4f |ICIR|>=%.4f, spearman<0.5) %.1fs =====" % (IC_GATE, ICIR_GATE, time.time() - t0))
for name, res in sorted(results.items()):
    ic, icir = res["ic"], res["icir"]
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    libok = res["max_abs_library_correlation"] is not None and res["max_abs_library_correlation"] < 0.5
    flag = "PASS" if (ok and libok) else ("GATE-OK-HI-CORR" if ok else "fail")
    dec = res["decay_ic_by_horizon"]
    print(f"{name:22s} IC={ic:+.4f} ICIR={icir:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov={res['coverage_asset_days']:.3f} n={res['n_ic_dates']} to={res['turnover_10d_rank']:.2f} "
          f"libcorr={res['max_abs_library_correlation']} decay{{1,5,10,20}}={dec['1']:+.3f}/{dec['5']:+.3f}/{dec['10']:+.3f}/{dec['20']:+.3f} -> {flag}")
