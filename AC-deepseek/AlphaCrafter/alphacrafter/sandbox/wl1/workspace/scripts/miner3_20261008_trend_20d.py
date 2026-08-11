"""miner_3 candidate: trend_20d (close vs 20d SMA distance).

Motivation: memory feedback (cycle4) asks to add a trend/momentum guard to the
reversal-heavy ensemble. Classic trend factor: price distance above/below its
20-day moving average. Positive values = uptrend, negative = downtrend.
"""
import sys
sys.path.insert(0, "scripts")
from miner3_20261008_lib import load_close_panel, run_validation

close = load_close_panel()
factor = close / close.rolling(20).mean() - 1.0

notes = ("Validated on 15-asset tradable universe, 2020-01..2026-07 (warm-up + "
         "online window to last completed day). Regimes: COVID crash, 2020-21 bull, "
         "2022 bear, 2023-24 AI rally, 2024-26 crypto/commodity cycles.")
run_validation(factor, close, factor_id="miner3_20261008_trend_20d",
               regime_notes=notes)
