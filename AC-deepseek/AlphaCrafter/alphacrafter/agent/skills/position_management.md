---
name: position_management
description: Determine full-investment target weights for the 15-asset long-only benchmark portfolio.
---

# Position Management Skill Document

## Regime assessment

Assess trend, volatility, liquidity, cross-asset correlation, breadth, and
sentiment using only data visible at the current decision date.

## Online portfolio contract

- The portfolio contains exactly the 15 `watch_list` assets.
- Target weights are non-negative and sum to 1.
- Online cash is zero; cash is not a sixteenth asset.
- Long-only: never open shorts.
- Fractional quantities are valid; do not enforce board lots.
- Bearish or high-risk views are expressed by tilting toward defensive
  tradable assets, not by reducing gross exposure to cash.
- If the target does not change, preserve the same complete target. Do not
  emit a cash-only or maintenance-only escape.

## Rebalance construction

1. Read the Screener's active factor ensemble and its normalized quality/IC
   tilt weights.
2. Compute a cross-sectional score for all 15 tradable assets, preserving each
   factor's direction.
3. Convert the score to a complete non-negative 15-asset target vector.
4. Submit the vector through the benchmark full-investment rebalance helper;
   the first allocation is free and later asset transfers cost 3 bps once.

The factor ensemble is capped at 10 active factors. The research library cap
and tail eviction are owned by the post-Miner factor contract, not by this
portfolio skill.

## Warm-up behavior

During shared warm-up, research and strategy registration are allowed, but the
Step tool must not advance the date or create holdings. The account remains at
1,000,000 initial cash until the first forward execution date, 2026-07-16.

## Validation

Before every online rebalance, verify the exact asset set, non-negative finite
weights, sum-to-one invariant, and absence of observation-only signal assets.
