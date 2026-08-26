"""miner_1 cycle 2031-06-12: explore correlation-structure reversal factor candidates.
Idea: after multi-day market stress, the cross-asset correlation structure snaps back;
assets whose trailing correlation with the market composite is EXTREME (high or low)
may exhibit reversal in forward returns -> captures crowding/de-crowding.

Validates ONE idea: cross-sectional "correlation-distance from median" factor.
Metrics: rank IC / ICIR at 10d horizon on the 15-asset tradable universe, 2020-03..2031-06-12.
"""