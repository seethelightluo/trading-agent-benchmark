"""miner_3 2035-06-29: Candidate 1 - Kaufman Efficiency Ratio (trend smoothness).
Idea: |close_t - close_{t-n}| / sum of |daily moves| over n days. High ER =
smooth directional trend; low ER = choppy range. Tests whether trend QUALITY
(not magnitude) predicts forward returns across the 15-asset panel.
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import ic_harness as H


def er_factor(panel, n):
    close = panel.astype(float)
    displacement = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return displacement / path


if __name__ == '__main__':
    panel = H.load_panel()
    print(f"Panel: {panel.shape[0]} dates x {panel.shape[1]} assets, "
          f"range {panel.index.min().date()}..{panel.index.max().date()}")
    for n in (10, 20, 60, 120):
        f = er_factor(panel, n)
        res = H.evaluate(f, f"ER_{n}d")
        H.print_results(f"ER_{n}d", res)
