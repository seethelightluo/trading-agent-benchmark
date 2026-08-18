"""miner_1 factor exploration screen - 2029-11-29 (optimized).

Screen candidate factors on the 15-instrument cross-asset universe.
Admission gates (10d horizon): |IC| >= 0.0070 and |ICIR| >= 0.0840.
IC = mean of daily cross-sectional Spearman corr(factor, fwd_10d_return).
ICIR = mean(IC)/std(IC). Require >=8 valid assets per date.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDJPY", "USDCNY", "EURUSD", "VIX"]
MIN_ASSETS = 8
HORIZON = 10


def load_asset(sym, days=2600):
    df = get_stock_daily_data(symbol=sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_macro(sym):
    df = pd.read_csv(f"../persistent/index_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def rolling_beta(x, y, win=60, minp=30):
    cov = x.rolling(win, min_periods=minp).cov(y)
    var = y.rolling(win, min_periods=minp).var()
    return cov / var


def compute_ic_metrics(factor_df, fwd_df):
    ic_list, dates = [], []
    vals = factor_df.values
    fwd = fwd_df.values
    idx = factor_df.index
    for i in range(len(idx)):
        fv = vals[i]
        fr = fwd[i]
        mask = ~(np.isnan(fv) | np.isnan(fr))
        if mask.sum() < MIN_ASSETS:
            continue
        rho = spearmanr(fv[mask], fr[mask])[0]
        if np.isfinite(rho):
            ic_list.append(rho)
            dates.append(idx[i])
    ic = np.array(ic_list)
    if len(ic) == 0:
        return None
    return {
        "ic": float(np.nanmean(ic)),
        "icir": float(np.nanmean(ic) / np.nanstd(ic)) if np.nanstd(ic) > 0 else 0.0,
        "ic_hit_ratio": float((ic > 0).mean()),
        "n_ic_dates": len(ic),
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
    }


def decay_by_horizon(factor_df, ret_df, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = ret_df.rolling(h).sum().shift(-h)
        m = compute_ic_metrics(factor_df, fwd)
        out[str(h)] = round(m["ic"], 4) if m else None
    return out


def turnover_10d(factor_df):
    rank_df = factor_df.rank(axis=1, pct=True)
    disp = (rank_df - rank_df.shift(HORIZON)).abs().mean(axis=1)
    return float(disp.dropna().mean())


def main():
    closes, highs, lows, rets = {}, {}, {}, {}
    for a in ASSETS:
        df = load_asset(a)
        if df is None:
            print(f"WARN no data {a}")
            continue
        closes[a] = df["close"]
        highs[a] = df["high"]
        lows[a] = df["low"]
        rets[a] = df["close"].pct_change()
    close_df = pd.DataFrame(closes).sort_index()
    high_df = pd.DataFrame(highs).sort_index()
    low_df = pd.DataFrame(lows).sort_index()
    ret_df = pd.DataFrame(rets).sort_index()
    print(f"price panel: {close_df.shape}, {close_df.index[0].date()}..{close_df.index[-1].date()}")

    macro_df = pd.DataFrame({m: load_macro(m)["close"] for m in MACRO}).sort_index()
    macro_ret = macro_df.pct_change()

    fwd_df = ret_df.rolling(HORIZON).sum().shift(-HORIZON)

    cand = {}
    cand["dxy_beta_60d_neg"] = -rolling_beta(ret_df, macro_ret["DXY"], 60, 30)
    cand["usdjpy_beta_60d_neg"] = -rolling_beta(ret_df, macro_ret["USDJPY"], 60, 30)
    cand["eurusd_beta_60d_neg"] = -rolling_beta(ret_df, macro_ret["EURUSD"], 60, 30)
    cand["usdcny_beta_60d_neg"] = -rolling_beta(ret_df, macro_ret["USDCNY"], 60, 30)
    cand["dd_dist_60d"] = close_df / close_df.rolling(60, min_periods=30).max() - 1.0
    rng20 = high_df.rolling(20, min_periods=10).max() - low_df.rolling(20, min_periods=10).min()
    cand["range_pos_20d"] = (close_df - low_df.rolling(20, min_periods=10).min()) / rng20 - 0.5
    intraday = ((high_df - low_df) / close_df).rolling(20, min_periods=10).mean()
    c2c = ret_df.rolling(20, min_periods=10).std()
    cand["park_ratio_20d_neg"] = -(intraday / c2c)
    cand["kurt_20d_neg"] = -ret_df.rolling(20, min_periods=10).kurt()
    # vectorized lag-1 autocorr over 20d
    r1 = ret_df.shift(1)
    cov = (ret_df * r1).rolling(20, min_periods=10).mean() - ret_df.rolling(20, min_periods=10).mean() * r1.rolling(20, min_periods=10).mean()
    v0 = ret_df.rolling(20, min_periods=10).var()
    v1 = r1.rolling(20, min_periods=10).var()
    cand["autocorr_20d"] = cov / np.sqrt(v0 * v1)
    cand["vol_ratio_5x60"] = ret_df.rolling(5, min_periods=3).std() / ret_df.rolling(60, min_periods=30).std()
    cand["hilo_range_60d"] = high_df.rolling(60, min_periods=30).max() / low_df.rolling(60, min_periods=30).min() - 1.0
    mom20 = close_df.shift(5) / close_df.shift(25) - 1.0
    cand["mom20_skip5_adj_vol"] = mom20 / ret_df.rolling(20, min_periods=10).std()
    bvix = rolling_beta(ret_df, macro_ret["VIX"], 60, 30)
    cand["vix_level_x_beta"] = bvix * macro_df["VIX"]

    print(f"\n{'factor':24s} {'IC':>8s} {'ICIR':>8s} {'hit':>6s} {'n_dates':>8s} {'covAD':>7s} {'covD8':>7s} {'turn10':>7s}")
    results = {}
    for name, f in cand.items():
        f = f.reindex(close_df.index)
        m = compute_ic_metrics(f, fwd_df)
        if m is None:
            print(f"{name:24s} NO VALID DATES")
            continue
        m["coverage_asset_days"] = float(f.notna().mean().mean())
        m["coverage_dates_ge8"] = float((f.notna().sum(axis=1) >= MIN_ASSETS).mean())
        m["turnover_10d_rank"] = turnover_10d(f)
        m["decay"] = decay_by_horizon(f, ret_df)
        results[name] = m
        flag = "  <-- GATE PASS" if (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084) else ""
        print(f"{name:24s} {m['ic']:8.4f} {m['icir']:8.4f} {m['ic_hit_ratio']:6.3f} {m['n_ic_dates']:8d} "
              f"{m['coverage_asset_days']:7.3f} {m['coverage_dates_ge8']:7.3f} {m['turnover_10d_rank']:7.3f}{flag}")

    print("\n--- negated direction check ---")
    for name, f in cand.items():
        f = f.reindex(close_df.index)
        m = compute_ic_metrics(-f, fwd_df)
        if m and abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084:
            print(f"negated {name:24s} IC={m['ic']:8.4f} ICIR={m['icir']:8.4f}  <-- GATE PASS (dir=-1)")

    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'decay'} | {'decay': v['decay']} for k, v in results.items()},
              open("scripts/miner1_20291129_screen_results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    import json
    main()
