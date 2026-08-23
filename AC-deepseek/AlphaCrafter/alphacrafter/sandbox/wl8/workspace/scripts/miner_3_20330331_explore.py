"""
miner_3 2033-03-31 exploration: fresh candidate factors on the 15-instrument
cross-asset tradable universe. All data observed through current date; no
lookahead (factor at t uses data <= t; forward return t..t+h).

Motivation: library is empty (all evicted/deprecated), trader runs fallback
mom10/vix-beta/yield-beta ensemble. Need robust, low-correlation factors with
|IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d admission horizon, stable across
regimes (2023-2033 includes vol spikes, crypto cycles, yield regime shifts).

Batch A candidates (interpretable, price/OHLC based):
 A) lr2_trend_60     : R^2 of log-price linear trend over 60d (trend certainty)
 B) gap_intensity_20 : mean |open/prev_close - 1| over 20d (overnight gap shock)
 C) serr_ac_20       : AR(1) coefficient of 1-day returns over 20d (trend persistence)
 D) close_loc_20     : mean (close-low)/(high-low) over 20d (close location / buying pressure)
 E) down_freq_60     : share of down days over 60d (drawdown frequency)
 F) updown_asym_20   : up-day mean ret - down-day mean ret (signed asymmetry)
 G) gk_vol_10x40     : Garman-Klass 10d vol / 40d vol (relative vol expansion)
 H) tail_shock_20    : max |1-day return| over 20d (tail shock recency)
 I) macd_rel_12x26   : MACD(12,26) histogram scaled by 60d mean (trend strength)
 J) wl_volume_20x60  : 20d/60d mean volume ratio (volume regime; secondary)
"""
import sys, os, json, base64, zlib, io
sys.path.insert(0,