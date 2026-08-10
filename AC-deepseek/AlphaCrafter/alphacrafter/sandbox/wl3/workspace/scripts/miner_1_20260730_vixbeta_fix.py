"""miner_1: fix calendar alignment for vix_beta_cond_60x20 re-validation.

VIX (and other macro series) trade on a sparser calendar than BTC/ETH/commodities.
Reindex the macro series onto each asset's own calendar with ffill so the rolling
60d beta computation sees valid pairs on every asset calendar date.
"""
import sys, importlib.util, warnings
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("common", "scripts/miner_1_20260730_common.py")
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
import pandas as pd
import numpy as np

closes, macros = common.load_all()


def f_vixbeta_aligned(c, a, macros):
    v = macros["VIX"]
    v = v.reindex(c.index).ffill()  # align macro onto asset calendar
    z = pd.concat([c.pct_change(), v.pct_change()], axis=1).dropna()
    if len(z) < 80:
        return None
    beta = c.pct_change().rolling(60).cov(v.pct_change()) / v.pct_change().rolling(60).var()
    out = -beta * (v / v.shift(20) - 1.0)
    return out.reindex(c.index)


res = common.run_full_validation(f_vixbeta_aligned, "vix_beta_cond_60x20", horizon=10,
                                 label="vix_beta_cond_60x20 (aligned)")
if res:
    ok = abs(res["ic"]) >= 0.007 and abs(res["icir"]) >= 0.084
    print(f"    => IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} "
          f"|gate| IC>=0.007 ICIR>=0.084 PASS={ok}")
    print("    metrics:", {k: v for k, v in res.items() if k != "decay_ic_by_horizon"})
    print("    decay:", res["decay_ic_by_horizon"])
