"""miner_1 cycle34: overnight vs intraday return decomposition factors.

Motivation: close-close momentum/vol factors dominate the library, but the
open->close vs close->open decomposition separates institutional/overnight
(Asian session reaction, US close news) from intraday speculative flows.
For 24/7 crypto, open ~= prev close so overnight ~= 0 -- this is itself
informative cross-sectionally (continuous vs discontinuous pricing).

Candidates:
  overnight_mom_20x5 : cumsum(log(1+overnight)) over 20d ending 5d ago
  intraday_mom_20x5  : cumsum(log(1+intraday)) over 20d ending 5d ago
  oi_spread_20x5     : overnight_mom - intraday_mom
  gap_share_60       : std(overnight)/std(close-close ret) over 60d
  oi_corr_60         : corr(overnight, close-close ret) over 60d
  overnight_share_20x5: |overnight_mom| / (|overnight_mom|+|intraday_mom|)
No lookahead: factors use data <= t; forward returns t+1..t+h.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_lib import (TRADABLES, VISIBLE_THROUGH, load_asset, load_panel,
                         forward_returns, compute_ic, coverage_stats,
                         turnover_rank, library_correlation, validate_factor)

pd.set_option("display.width", 200)


def load_ohlc_panel():
    frames = {}
    for a in TRADABLES:
        df = load_asset(a)
        df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        frames[a] = df[["open", "close"]]
    return frames


def build_factors(frames, panel):
    """Return dict of candidate factor panels (union-indexed)."""
    out = {}
    for a, df in frames.items():
        o = df["open"]
        c = df["close"]
        prev_c = c.shift(1)
        ovn = (o / prev_c - 1.0)          # overnight return on own calendar
        intra = (c / o - 1.0)             # intraday return on own calendar
        tot = c.pct_change()              # close-close return
        # --- momentum of components (20d window, skip 5) ---
        ovn_mom = np.log1p(ovn).rolling(20).sum().shift(5)
        intra_mom = np.log1p(intra).rolling(20).sum().shift(5)
        out.setdefault("overnight_mom_20x5", {})[a] = ovn_mom
        out.setdefault("intraday_mom_20x5", {})[a] = intra_mom
        out.setdefault("oi_spread_20x5", {})[a] = ovn_mom - intra_mom
        aovn = ovn_mom.abs()
        aint = intra_mom.abs()
        out.setdefault("overnight_share_20x5", {})[a] = aovn / (aovn + aint)
        # --- gap share of volatility (60d) ---
        out.setdefault("gap_share_60", {})[a] = ovn.rolling(60).std() / tot.rolling(60).std()
        # --- correlation of overnight with total return (60d) ---
        out.setdefault("oi_corr_60", {})[a] = ovn.rolling(60).corr(tot)
    return {k: pd.DataFrame(v, index=panel.index) for k, v in out.items()}


def main():
    print("=" * 100)
    print("CYCLE 34: OVERNIGHT/INTRADAY DECOMPOSITION FAMILY")
    print("Visible through:", VISIBLE_THROUGH)
    print("=" * 100)
    panel = load_panel()
    print("panel shape:", panel.shape, "| assets:", len(TRADABLES))
    frames = load_ohlc_panel()
    cands = build_factors(frames, panel)

    # Library correlation reference (recompute core library signals)
    lib = __import__("miner_1_lib")
    library = lib.load_library_signals(panel)

    fwd_cache = {}
    for h in (1, 2, 3, 5, 10, 20):
        fwd_cache[str(h)] = forward_returns(panel, h)

    print("\n%-26s %7s %7s %6s %7s %8s %8s %8s %8s" % (
        "factor", "IC", "ICIR", "hit", "n", "cov_asset", "turn10",
        "maxlib", "PASS"))
    results = {}
    for name, sig in cands.items():
        m = validate_factor(sig, panel, library=library, fwd_cache=fwd_cache)
        to = turnover_rank(sig, step=10)
        m["turnover_10d_rank"] = round(to, 4) if to == to else None
        ic, icir = abs(m["ic"]), abs(m["icir"])
        passed = (ic >= 0.007) and (icir >= 0.084)
        results[name] = {"metrics": m, "pass": passed}
        print("%-26s %7.4f %7.4f %6.3f %7d %8.3f %8.3f %8.3f %8s" % (
            name, m["ic"], m["icir"], m["ic_hit_ratio"], m["n_ic_dates"],
            m["coverage_asset_days"], m["turnover_10d_rank"],
            m.get("max_abs_library_correlation", float("nan")),
            "PASS" if passed else "FAIL"))

    # --- deep dive on the two most promising: yearly regime IC ---
    print("\n" + "=" * 100)
    print("REGIME STABILITY (yearly IC of IC series at horizon 10)")
    ret10 = fwd_cache["10"]
    for name in ["overnight_mom_20x5", "intraday_mom_20x5", "oi_spread_20x5",
                 "gap_share_60", "oi_corr_60"]:
        sig = cands[name]
        ic_ser = compute_ic(sig, ret10, 8).dropna()
        years = ic_ser.index.year
        parts = []
        for y in sorted(set(years)):
            sub = ic_ser[years == y]
            parts.append("%s:%.4f/%.3f(n%d)" % (y, sub.mean(),
                         (sub.mean()/sub.std()) if sub.std() > 0 else 0.0, len(sub)))
        print("%-24s full=%.4f | %s" % (name, ic_ser.mean(), " ".join(parts)))

    import json
    with open("scripts/_miner1_cycle34_overnight_results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\nresults saved.")


if __name__ == "__main__":
    main()
