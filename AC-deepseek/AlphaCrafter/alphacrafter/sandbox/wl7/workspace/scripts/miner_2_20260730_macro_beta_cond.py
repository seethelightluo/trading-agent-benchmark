"""miner_2 factor family #1: conditional macro-beta risk factors.
Idea: an asset's sensitivity to a macro driver (DXY / USDJPY / EURUSD / USDCNY),
interacted with the recent macro move, identifies directional risk-hedge
opportunities. Analogous to vix_beta_cond_60x20 but with FX macro drivers.
factor = -beta(asset_ret, macro_ret, B) * (macro_t / macro_{t-B2} - 1)
One idea per script: macro-beta conditional family (parameter sweep).
"""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from miner_2_lib import validate_factor, per_asset, load_panel, load_macro

MACROS = ["DXY", "USDJPY", "EURUSD", "USDCNY"]


def make_macro_beta(macro: str, beta_win: int, macro_win: int):
    def fn(panel, macro_dict):
        m = macro_dict[macro]
        mr = m.pct_change()
        def f(s):
            r = s.pct_change()
            z = pd.concat([r.rename("r"), mr.reindex(s.index).rename("m")], axis=1)
            beta = z["r"].rolling(beta_win).cov(z["m"]) / z["m"].rolling(beta_win).var().replace(0, pd.NA)
            mv = (m / m.shift(macro_win) - 1.0).reindex(s.index)
            return -beta * mv
        return per_asset(f)(panel, macro_dict)
    return fn


if __name__ == "__main__":
    panel = load_panel()
    macro = load_macro()
    print(f"panel: {panel.index[0].date()} .. {panel.index[-1].date()}, assets={panel.shape[1]}")
    for m in MACROS:
        for bw, mw in ((30, 10), (60, 20), (60, 60), (90, 20)):
            validate_factor(f"macrob_beta{m}_{bw}x{mw}",
                            make_macro_beta(m, bw, mw))
