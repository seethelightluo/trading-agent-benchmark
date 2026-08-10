"""miner_1 re-validation of existing library factors through 2026-07-29."""
import sys, importlib.util
spec = importlib.util.spec_from_file_location("common", "scripts/miner_1_20260730_common.py")
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
import pandas as pd


def f_mom10(c, a, macros):
    return c.shift(5) / c.shift(15) - 1.0

def f_mom120(c, a, macros):
    return c.shift(5) / c.shift(125) - 1.0

def f_vixbeta(c, a, macros):
    v = macros["VIX"]
    z = pd.concat([c.pct_change(), v.pct_change()], axis=1).dropna()
    if len(z) < 80:
        return None
    r = c.pct_change()
    beta = r.rolling(60).cov(v.pct_change()) / v.pct_change().rolling(60).var()
    out = -beta * (v / v.shift(20) - 1.0)
    return out

def f_volvol(c, a, macros):
    rv = c.pct_change().rolling(20).std()
    return rv.rolling(60).std()

print("=== Re-validation through 2026-07-29 (was 2020-01-01..2026-07-15) ===")
for fid, fn in [("mom_10d_skip5", f_mom10), ("mom_120d_skip5", f_mom120),
                ("vix_beta_cond_60x20", f_vixbeta), ("vol_of_vol20x60", f_volvol)]:
    print("-" * 70)
    res = common.run_full_validation(fn, fid, horizon=10, label=fid)
    if res:
        print(f"    => IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} |gate| IC>=0.007 ICIR>=0.084 "
              f"PASS={abs(res['ic'])>=0.007 and abs(res['icir'])>=0.084}")
