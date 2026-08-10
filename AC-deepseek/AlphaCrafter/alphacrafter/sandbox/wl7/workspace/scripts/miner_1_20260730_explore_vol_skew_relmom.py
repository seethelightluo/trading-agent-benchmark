"""
miner_1 exploration: candle/range + volatility/skewness + relative-momentum families.
Each candidate evaluated independently at horizon 10 on 15-asset universe.
"""
import pandas as pd, numpy as np, json, sys
sys.path.insert(0, "scripts")
from miner_1_metrics import (load_panel, panel_col, evaluate, gate_pass,
                             library_corr, library_signal)

def main():
    frames = load_panel()
    closes = panel_col(frames, "close")
    opens  = panel_col(frames, "open")
    highs  = panel_col(frames, "high")
    lows   = panel_col(frames, "low")
    rets   = closes.pct_change()
    print(f"panel: {closes.index[0].date()} .. {closes.index[-1].date()}, assets={closes.shape[1]}")

    cands = {}
    # ---- candle/range family ----
    hl = (highs - lows).replace(0, np.nan)
    cands["close_pos_20d"] = ((closes - lows) / hl).rolling(20).mean()
    cands["close_pos_5d"]  = ((closes - lows) / hl).rolling(5).mean()
    body = (closes - opens).abs()
    cands["body_ratio_20d"] = (body / hl).rolling(20).mean()
    shadow = (highs - pd.concat([opens, closes], axis=1).max(axis=1)) / hl
    cands["upper_shadow_20d"] = shadow.rolling(20).mean()
    cands["range_20d"] = (hl / closes).rolling(20).mean()
    prev_close = closes.shift(1)
    overnight = (opens - prev_close) / prev_close
    cands["overnight_20d"] = overnight.rolling(20).mean()
    intraday = (closes - opens) / opens
    cands["intraday_20d"] = intraday.rolling(20).mean()
    # gap + intraday momentum: today's candle position interacted with gap
    # ---- volatility / skew family ----
    cands["vol_20d"] = rets.rolling(20).std()
    cands["vol_60d"] = rets.rolling(60).std()
    cands["vol_ratio_5x60"] = rets.rolling(5).std() / rets.rolling(60).std()
    cands["downside_vol_60d"] = rets.clip(upper=0).rolling(60).std()
    cands["skew_60d"] = rets.rolling(60).skew()
    cands["kurt_60d"] = rets.rolling(60).kurt()
    cands["max_dd_60d"] = closes / closes.rolling(60).max() - 1.0
    # tail ratio: fraction of large negative vs positive moves over 60d
    neg_frac = (rets < -2 * rets.rolling(60).std()).rolling(60).mean()
    pos_frac = (rets > 2 * rets.rolling(60).std()).rolling(60).mean()
    cands["tail_ratio_60d"] = neg_frac / pos_frac.replace(0, np.nan)
    # ---- relative momentum family ----
    mkt = closes.pct_change().mean(axis=1)
    rel_20 = (closes / closes.shift(20) - 1.0).sub(mkt.rolling(20).mean(), axis=0)
    rel_60 = (closes / closes.shift(60) - 1.0).sub(mkt.rolling(60).mean(), axis=0)
    cands["rel_mom_20d"] = rel_20
    cands["rel_mom_60d"] = rel_60
    # correlation with market (diversifier)
    mkt_ret = closes.pct_change().mean(axis=1)
    cands["corr_mkt_60d"] = rets.rolling(60).corr(mkt_ret)
    # beta to market over 60d
    var_m = mkt_ret.rolling(60).var()
    cands["beta_mkt_60d"] = rets.rolling(60).cov(mkt_ret) / var_m.replace(0, np.nan)

    lib_ids = ["mom_10d_skip5", "mom_120d_skip5", "vol_of_vol20x60", "vix_beta_cond_60x20"]
    results = []
    for fid, fv in cands.items():
        res = evaluate(fv, closes, horizon=10, label=fid)
        if res is None:
            continue
        max_corr, per = library_corr(fv, closes, lib_ids)
        res["max_abs_library_correlation"] = max_corr
        res["per_factor_corr"] = per
        results.append((fid, res))
        flag = "PASS" if gate_pass(res) else "fail"
        print(f"[{flag}] {fid}: ic={res['ic']} icir={res['icir']} hit={res['ic_hit_ratio']} "
              f"n={res['n_ic_dates']} cov_ad={res['coverage_asset_days']} "
              f"turn={res['turnover_10d_rank']} maxcorr={max_corr}")
        print("   decay:", res["decay_ic_by_horizon"])

    print("\n=== SUMMARY: PASS GATE @h10 ===")
    for fid, res in results:
        if gate_pass(res):
            print(f"PASS {fid}: ic={res['ic']} icir={res['icir']} maxcorr={res['max_abs_library_correlation']}")
    with open("scripts/_vol_candle_results.json", "w") as f:
        json.dump([{"factor_id": k, **v} for k, v in results], f, indent=1, default=str)

if __name__ == "__main__":
    main()
