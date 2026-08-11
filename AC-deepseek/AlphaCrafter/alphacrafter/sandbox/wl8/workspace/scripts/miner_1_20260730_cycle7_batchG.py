"""miner_1 2026-07-30 cycle 7 batch G - NEW factor ideas not previously explored.
Motivations:
 - Library is mom_10d_skip5 / vix_beta_cond_60x20 / yield_beta_cond_60x20.
 - Eviction gate: pairwise |rho| > 0.5 vs library panels (deterministic recompute).
 - Explore orthogonal families: vol-adjusted trend, MA gradient, variance-ratio
   (serial correlation), momentum acceleration, USDCNY/DXY macro-conditional beta,
   OHLC shadow structure, volume trend, gap behavior, trend-gated momentum,
   cross-sectional relative momentum.
Validation: cross-sectional rank IC/ICIR at horizon 10 on the 15-asset universe,
data through 2026-07-29 (previous completed trading day). Gates |IC|>=0.007, |ICIR|>=0.084.
"""
import sys, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (ASSETS, DATA_DIR, INDEX_DIR, load_closes, load_index,
                                   factor_panel, fwd_returns, ic_series, turnover_rank,
                                   coverage, IC_GATE, ICIR_GATE, MIN_ASSETS_PER_DATE)

END = pd.Timestamp("2026-07-29")
IC_HORIZON = 10

close, vol, open_, high, low = load_closes(end_date=END)
macro = {
    "VIX": load_index("VIX").loc[:END],
    "DXY": load_index("DXY").loc[:END],
    "USDCNY": load_index("USDCNY").loc[:END],
    "USDJPY": load_index("USDJPY").loc[:END],
    "EURUSD": load_index("EURUSD").loc[:END],
}
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")


def _beta_cond(c, m, beta_win, move_win, sign=-1.0):
    r = c.pct_change()
    mr = m.pct_change()
    beta = r.rolling(beta_win).cov(mr) / mr.rolling(beta_win).var()
    mv = m / m.shift(move_win) - 1.0
    return sign * beta * mv


def f_vol_adj_mom_10x20(c, v, o, h, l, m):
    mom = c.shift(5) / c.shift(15) - 1.0
    sd = c.pct_change().rolling(20).std()
    return mom / sd


def f_ma_grad_20x60(c, v, o, h, l, m):
    return c.rolling(20).mean() / c.rolling(60).mean() - 1.0


def f_var_ratio_5x20(c, v, o, h, l, m):
    r = c.pct_change()
    var1 = r.rolling(20).var()
    var5 = r.rolling(5).sum().rolling(40).var()
    return var5 / (5.0 * var1)


def f_ret_ac_10(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(10).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=True)


def f_accel_20x10(c, v, o, h, l, m):
    mom20 = c / c.shift(20) - 1.0
    return mom20 - mom20.shift(10)


def f_usdcny_beta_cond_60x20(c, v, o, h, l, m):
    return _beta_cond(c, m["USDCNY"], 60, 20, sign=-1.0)


def f_dxy_beta_cond_30x10(c, v, o, h, l, m):
    return _beta_cond(c, m["DXY"], 30, 10, sign=-1.0)


def f_range_eff_20(c, v, o, h, l, m):
    rng = ((h - l) / c).rolling(20).mean()
    sd = c.pct_change().rolling(20).std()
    return rng / sd


def f_up_shadow_20(c, v, o, h, l, m):
    return (h / np.maximum(o, c) - 1.0).rolling(20).mean()


def f_dn_shadow_20(c, v, o, h, l, m):
    return (np.minimum(o, c) / l - 1.0).rolling(20).mean()


def f_vol_trend_5x20(c, v, o, h, l, m):
    return v.rolling(5).mean() / v.rolling(20).mean()


def f_gap_ratio_20(c, v, o, h, l, m):
    return (o / c.shift(1) - 1.0).rolling(20).mean()


def f_trend_gate_mom_10(c, v, o, h, l, m):
    mom = c.shift(5) / c.shift(15) - 1.0
    gate = np.sign(c.rolling(20).mean() - c.rolling(60).mean())
    return mom * gate


def f_rel_mom_cs_20(c, v, o, h, l, m):
    mom20 = c / c.shift(20) - 1.0
    med = mom20.rolling(1).median()
    return mom20 - med


def f_hl_amp_20(c, v, o, h, l, m):
    return ((h - l) / c).rolling(20).mean()


CANDIDATES = [
    ("vol_adj_mom_10x20", f_vol_adj_mom_10x20, "vol-scaled short momentum (t-stat of 10d trend)"),
    ("ma_grad_20x60", f_ma_grad_20x60, "MA20/MA60 gradient (trend slope)"),
    ("var_ratio_5x20", f_var_ratio_5x20, "variance ratio: trend vs mean-reversion regime"),
    ("ret_ac_10", f_ret_ac_10, "1-lag autocorrelation of daily returns (10d)"),
    ("accel_20x10", f_accel_20x10, "momentum acceleration (change in 20d return)"),
    ("usdcny_beta_cond_60x20", f_usdcny_beta_cond_60x20, "conditional USDCNY beta x 20d CNY move"),
    ("dxy_beta_cond_30x10", f_dxy_beta_cond_30x10, "conditional DXY beta (short windows) x 10d DXY move"),
    ("range_eff_20", f_range_eff_20, "range/vol efficiency ratio (20d)"),
    ("up_shadow_20", f_up_shadow_20, "mean upper candle shadow ratio (20d)"),
    ("dn_shadow_20", f_dn_shadow_20, "mean lower candle shadow ratio (20d)"),
    ("vol_trend_5x20", f_vol_trend_5x20, "volume trend SMA5/SMA20"),
    ("gap_ratio_20", f_gap_ratio_20, "mean overnight gap (20d)"),
    ("trend_gate_mom_10", f_trend_gate_mom_10, "trend-gated short momentum (interaction)"),
    ("rel_mom_cs_20", f_rel_mom_cs_20, "cross-sectional relative 20d momentum"),
    ("hl_amp_20", f_hl_amp_20, "mean daily high-low amplitude (20d)"),
]


def load_effective_library():
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        try:
            d = json.load(open(f"factors/{fid}.json"))
            data = d["validation"]["signal_artifact"]["data"]
            raw = base64.b64decode(data)
            csv_text = zlib.decompress(raw).decode()
            panel = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=True)
            panel.index = pd.DatetimeIndex(panel.index)
            panel = panel.loc[panel.index <= END]
            lib[fid] = panel
        except Exception as e:
            print(f"[warn] library {fid}: {e}")
    return lib


LIB = load_effective_library()
print(f"Library panels loaded: {list(LIB.keys())}")


def max_library_corr(panel, lib):
    out = {}
    for fid, lp in lib.items():
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
        out[fid] = float(np.corrcoef(a[m], b[m])[0, 1])
    vals = [abs(x) for x in out.values() if np.isfinite(x)]
    return (max(vals) if vals else np.nan), out


def evaluate(name, fn):
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    cov_ad, cov_ge8 = coverage(panel)
    fr = fwd_returns(close, IC_HORIZON)
    ic = ic_series(panel, fr)
    n = len(ic)
    if n < 30:
        print(f"{name}: degenerate n_ic_dates={n} cov={cov_ad:.3f}")
        return None
    icv = float(ic.mean())
    icir = float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0
    hit = float((ic > 0).mean()) if icv >= 0 else float((ic < 0).mean())
    to = turnover_rank(panel)
    maxrho, rhomap = max_library_corr(panel, LIB)
    gate = abs(icv) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"{name:24s} IC={icv:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={n} "
          f"cov_ad={cov_ad:.3f} cov8={cov_ge8:.3f} to={to:.2f} "
          f"libcorr={ {k: (round(v,3) if np.isfinite(v) else None) for k,v in rhomap.items()} } "
          f"max={maxrho if np.isfinite(maxrho) else None:.4f} -> {'PASS' if gate else 'fail'}")
    return dict(name=name, ic=icv, icir=icir, hit=hit, n=n, cov_ad=cov_ad, cov8=cov_ge8,
                to=to, maxrho=(round(maxrho, 4) if np.isfinite(maxrho) else None), rhomap=rhomap,
                panel=panel, gate=gate)


results = {}
for name, fn, desc in CANDIDATES:
    try:
        r = evaluate(name, fn)
        if r:
            results[name] = r
    except Exception as e:
        print(f"{name}: ERROR {e}")

print("\n===== SUMMARY =====")
for name, r in results.items():
    print(f"{name:24s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} n={r['n']} cov={r['cov_ad']:.3f} "
          f"to={r['to']:.2f} maxrho={r['maxrho']} -> {'PASS' if r['gate'] else 'fail'}")

with open("scripts/_miner1_cycle7_batchG_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "panel"} for k, v in results.items()},
              f, indent=1, default=str)
print("[saved] scripts/_miner1_cycle7_batchG_results.json")
