"""miner_1 cycle34b: deep validation of intraday_mom_20x5 (candidate winner)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import json
from miner_1_lib import (TRADABLES, VISIBLE_THROUGH, load_asset, load_panel,
                         forward_returns, compute_ic, coverage_stats,
                         turnover_rank, library_correlation, validate_factor,
                         panel_rank_corr)

pd.set_option("display.width", 220)


def load_ohlc_panel():
    frames = {}
    for a in TRADABLES:
        df = load_asset(a)
        df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        frames[a] = df[["open", "close"]]
    return frames


def intraday_mom_panel(frames, panel, win=20, skip=5):
    out = {}
    for a, df in frames.items():
        o, c = df["open"], df["close"]
        intra = c / o - 1.0
        out[a] = np.log1p(intra).rolling(win).sum().shift(skip).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def oi_spread_panel(frames, panel, win=20, skip=5):
    out = {}
    for a, df in frames.items():
        o, c = df["open"], df["close"]
        ovn = o / c.shift(1) - 1.0
        intra = c / o - 1.0
        s = np.log1p(intra).rolling(win).sum().shift(skip) - \
            np.log1p(ovn).rolling(win).sum().shift(skip)
        out[a] = s.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def main():
    print("=" * 100)
    print("CYCLE 34b DEEP VALIDATION: intraday_mom_20x5")
    panel = load_panel()
    frames = load_ohlc_panel()
    lib = __import__("miner_1_lib")
    library = lib.load_library_signals(panel)

    cand = intraday_mom_panel(frames, panel)
    spread = oi_spread_panel(frames, panel)

    fwd_cache = {}
    for h in (1, 2, 3, 5, 10, 20):
        fwd_cache[str(h)] = forward_returns(panel, h)
    ret10 = fwd_cache["10"]

    m = validate_factor(cand, panel, library=library, fwd_cache=fwd_cache)
    to = turnover_rank(cand, step=10)
    print("FULL-SAMPLE VALIDATION (horizon 10):")
    print(json.dumps({k: m[k] for k in m if k != "library_pairwise_corr"},
                     indent=1, default=str))
    print("\nLIBRARY PAIRWISE (ranked mean daily):")
    for fid, r in sorted(m["library_pairwise_corr"].items(), key=lambda kv: -abs(kv[1])):
        print("   %-24s %+.4f" % (fid, r))

    # correlation between the two passing candidates
    rho = panel_rank_corr(cand, spread)
    print("\npanel rank corr intraday_mom_20x5 vs oi_spread_20x5: %.4f" % rho)

    # regime split
    ic_ser = compute_ic(cand, ret10, 8).dropna()
    years = ic_ser.index.year
    print("\nYEARLY IC:")
    for y in sorted(set(years)):
        sub = ic_ser[years == y]
        print("   %s: ic=%.4f icir=%.3f n=%d" % (y, sub.mean(),
              (sub.mean()/sub.std()) if sub.std() > 0 else 0.0, len(sub)))
    for lbl, n in [("last252", 252), ("last126", 126), ("last63", 63)]:
        sub = ic_ser.iloc[-n:]
        print("   %s: ic=%.4f icir=%.3f hit=%.3f n=%d" % (
            lbl, sub.mean(), (sub.mean()/sub.std()) if sub.std() > 0 else 0.0,
            (np.sign(sub) == np.sign(sub.mean())).mean(), len(sub)))

    # horizon robustness of intraday_mom with skip variants
    print("\nPARAM SWEEP (win, skip) -> IC / ICIR:")
    for (w, s) in [(10, 3), (20, 5), (30, 5), (20, 10), (40, 5), (60, 5)]:
        sig = intraday_mom_panel(frames, panel, w, s)
        mm = validate_factor(sig, panel, library=library, fwd_cache=fwd_cache)
        print("   win=%2d skip=%2d: IC=%+.4f ICIR=%+.4f maxlib=%.3f turn10=%.3f" % (
            w, s, mm["ic"], mm["icir"],
            mm.get("max_abs_library_correlation", float("nan")),
            turnover_rank(sig, step=10)))


if __name__ == "__main__":
    main()
