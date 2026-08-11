"""miner_2 fast validation runner (per-asset calendar-aware) with cached libs.
Usage: python scripts/miner_2_run_fast.py <candidate_key>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
import miner_2_lib as lib

EPS = 1e-12
WATCH = lib.WATCH
MAX_VISIBLE = lib.MAX_VISIBLE
FACTOR_LAST = lib.FACTOR_LAST
MIN_ASSETS = lib.MIN_ASSETS
ADMISSION = lib.ADMISSION

panel = lib.load_panel()
rets = panel.pct_change()
mac = lib.load_macro()
libs = lib.library_signals(panel)


def per_asset_series(fn):
    """fn(c: pd.Series) -> pd.Series on asset's own calendar, reindexed to union."""
    out = {}
    for s in WATCH:
        c = panel[s].dropna()
        out[s] = fn(c)
    return pd.DataFrame(out, index=panel.index)


def per_asset_ohlcv(fn):
    """fn(df) -> pd.Series on asset's own calendar using full OHLCV frame."""
    out = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[s] = fn(df)
    return pd.DataFrame(out, index=panel.index)


def skew_20():
    return per_asset_series(lambda c: c.pct_change().rolling(20, min_periods=10).skew())


def drawdown_60():
    return per_asset_series(lambda c: c / c.rolling(60, min_periods=10).max() - 1.0)


def time_since_high_60():
    def f(c):
        rmax = c.rolling(60, min_periods=10).max()
        hit = (c >= rmax).astype(float)
        groups = (hit == 0).cumsum()
        days = groups.groupby(groups).cumcount() + 1
        return days.where(hit == 0, 0.0)
    return per_asset_series(f)


def amihud_20():
    def f(df):
        r = df["close"].pct_change()
        return (r.abs() / (df["volume"] + EPS)).rolling(20, min_periods=10).mean()
    return per_asset_ohlcv(f)


def vol_ratio_10x60():
    return per_asset_series(lambda c: (c.pct_change().rolling(10, min_periods=5).std()
                                       / c.pct_change().rolling(60, min_periods=20).std()))


def usdjpy_beta_cond_60x20():
    usdjpy = mac["USDJPY"]
    usdjpyr = usdjpy.pct_change()
    beta = rets.rolling(60, min_periods=20).cov(usdjpyr) / usdjpyr.rolling(60, min_periods=20).var()
    cond = usdjpy / usdjpy.shift(20) - 1.0
    return beta * cond


def dxy_beta_cond_60x20():
    dxy = mac["DXY"]
    dxy_r = dxy.pct_change()
    beta = rets.rolling(60, min_periods=20).cov(dxy_r) / dxy_r.rolling(60, min_periods=20).var()
    cond = dxy / dxy.shift(20) - 1.0
    return beta * cond


def vol_trend(short, long):
    def f(df):
        v = df["volume"]
        return np.log((v.rolling(short, min_periods=max(2, short // 2)).mean() + EPS)
                      / (v.rolling(long, min_periods=max(5, long // 2)).mean() + EPS))
    return per_asset_ohlcv(f)


def body_ratio(w):
    def f(df):
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        body = (df["close"] - df["open"]).abs() / (rng + EPS)
        return body.rolling(w, min_periods=max(5, w // 2)).mean()
    return per_asset_ohlcv(f)


def upper_wick(w):
    def f(df):
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        uw = (df["high"] - np.maximum(df["open"], df["close"])) / (rng + EPS)
        return uw.rolling(w, min_periods=max(5, w // 2)).mean()
    return per_asset_ohlcv(f)


def range_pos(w):
    def f(df):
        rng = df["high"].rolling(w, min_periods=max(5, w // 2)).max() \
            - df["low"].rolling(w, min_periods=max(5, w // 2)).min()
        return (df["close"] - df["low"].rolling(w, min_periods=max(5, w // 2)).min()) / (rng + EPS)
    return per_asset_ohlcv(f)


def vol_confirm_mom10():
    def f(df):
        mom = df["close"] / df["close"].shift(10) - 1.0
        vt = np.log((df["volume"].rolling(5, min_periods=3).mean() + EPS)
                    / (df["volume"].rolling(60, min_periods=30).mean() + EPS))
        return mom * np.sign(vt)
    return per_asset_ohlcv(f)


CANDIDATES = {
    "skew_20": skew_20,
    "drawdown_60": drawdown_60,
    "time_since_high_60": time_since_high_60,
    "amihud_20": amihud_20,
    "vol_ratio_10x60": vol_ratio_10x60,
    "usdjpy_beta_cond_60x20": usdjpy_beta_cond_60x20,
    "dxy_beta_cond_60x20": dxy_beta_cond_60x20,
    "vol_trend_5x60": lambda: vol_trend(5, 60),
    "body_ratio_20": lambda: body_ratio(20),
    "upper_wick_20": lambda: upper_wick(20),
    "range_pos_20": lambda: range_pos(20),
    "vol_confirm_mom10": vol_confirm_mom10,
}


def fwd_returns(h):
    out = {}
    for s in WATCH:
        c = panel[s].dropna()
        out[s] = c.shift(-h) / c - 1.0
    return pd.DataFrame(out, index=panel.index)


def rank_ic(factor, fwd):
    ics = {}
    idx = factor.index.intersection(fwd.index)
    for d in idx:
        f = factor.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        if len(r) < MIN_ASSETS:
            continue
        ics[d] = spearmanr(f.reindex(r.index), r)[0]
    return pd.Series(ics).sort_index()


def library_corr_fast(factor, window=350):
    per = {}
    common = factor.index.intersection(panel.index)[-window:]
    for fid, lf in libs.items():
        cs = []
        for dt in common:
            if dt not in factor.index or dt not in lf.index:
                continue
            f = factor.loc[dt]
            g = lf.loc[dt]
            m = f.notna() & g.notna()
            m = m.reindex(f.index).fillna(False)
            if int(m.sum()) >= MIN_ASSETS:
                cs.append(spearmanr(f[m], g[m])[0])
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


def turnover_10d(factor):
    ranks = factor.rank(axis=1)
    out = []
    for i in range(10, len(ranks)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) < MIN_ASSETS:
            continue
        out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def run(name, direction_override=None, horizons=(1, 2, 3, 5, 10, 20)):
    factor = CANDIDATES[name]()
    factor_w = factor.loc[:FACTOR_LAST]
    res = {"name": name, "factor_rows": len(factor_w), "n_assets": panel.shape[1]}
    fwd = {h: fwd_returns(h) for h in horizons}
    ic_by_h = {h: rank_ic(factor_w, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = direction_override if direction_override is not None else (
        float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0)
    ic_by_h = {h: ic * direction for h, ic in ic_by_h.items()}
    for h in horizons:
        ic = ic_by_h[h]
        icir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = icir
        res[f"hit_h{h}"] = float((ic > 0).mean()) if len(ic) else float("nan")
        res[f"n_dates_h{h}"] = int(len(ic))
    res["direction"] = direction
    valid = factor_w.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d(factor_w)
    max_corr, per = library_corr_fast(factor_w)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in horizons}
    gate_ic = abs(res["ic_h10"]) >= ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= ADMISSION["icir"]
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "pass": bool(gate_ic and gate_icir)}
    print(f"=== {name} ===")
    print(f"  window: {factor_w.index.min().date()} .. {factor_w.index.max().date()}, "
          f"{len(factor_w)} dates, {panel.shape[1]} assets")
    print(f"  direction={direction:+.3f}")
    for h in horizons:
        print(f"  h{h:>2}: IC={res[f'ic_h{h}']:+.4f}  ICIR={res[f'icir_h{h}']:+.4f}  "
              f"hit={res[f'hit_h{h}']:.3f}  n={res[f'n_dates_h{h}']}")
    print(f"  coverage_asset_days={res['coverage_asset_days']:.3f}  "
          f"coverage_dates_ge8={res['coverage_dates_ge8']:.3f}  "
          f"turnover_10d_rank={res['turnover_10d_rank']:.3f}")
    print(f"  max_abs_library_corr={max_corr:.3f}  per={per}")
    print(f"  ADMISSION (h=10): |IC|={abs(res['ic_h10']):.4f} (>=0.007: {gate_ic}), "
          f"|ICIR|={abs(res['icir_h10']):.4f} (>=0.084: {gate_icir}) -> "
          f"{'PASS' if gate_ic and gate_icir else 'FAIL'}")
    print()
    return res


if __name__ == "__main__":
    run(sys.argv[1])
