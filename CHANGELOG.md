# Changelog

## 0.2.0 - 2026-08-07

- Add expected normalized finishing performance (`xP`) and exact driver/car/team
  Shapley attribution alongside `xW`.
- Restore position, teammate, and failure-aware driver-rating signals from the legacy
  F1 research scripts.
- Add recency-weighted circuit passability from prior same-circuit races.
- Add cached ERA5 historical race-window weather, actual race distance, and regulation
  era context.
- Support one cross-fitted timeline from 1950 through 2025, including shared drives and
  fractional shared-win credit.
- Add interactive race-by-race rating-history exports.
- Validate a complete 76-season run covering 1,149 races and 25,784 driver-race entries.

## 0.1.0 - 2026-07-17

- Initial Historical XW driver/car/team decomposition and first-stage ranking prototype.
