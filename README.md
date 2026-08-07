# Historical XW

Current version: **0.2.0**

Historical XW is a historical-only Formula 1 analytics package that attributes a completed race's field-normalized win probability to driver (`XdW`), technical car (`XcW`), and team operations (`XtW`). It never treats those values as three independent win probabilities. Instead, it evaluates all eight actual/neutral component coalitions and uses exact grouped Shapley values, preserving nonlinear interactions.

It now also includes the first combined driver-ranking model derived from the legacy
`E:\F1` Elo research. The race update is `K * (observed_win - XW)`: an expected win
still increases a driver's rating, while an unlikely win earns a larger increase. The
maintained version also restores the legacy position signal as `XP`, expected normalized
finishing performance from `1` (winner) to `0` (last).
See [the ranking architecture](docs/ranking-architecture.md) for the model diagram,
legacy mapping, equations, and current limitations.

## What is runnable now

- Exact, independently stored XdW/XcW/XtW scores and probability-point contributions.
- Neutral driver, car, team, and all-neutral counterfactuals.
- Direct replacement uplifts and exact Shapley efficiency checks.
- Field-normalized historical win probabilities and uncertainty from Monte Carlo capability sampling.
- Five-metre canonical circuit reconstruction from multiple clean telemetry laps.
- Parquet output and a queryable DuckDB view.
- Historical cache/source registry for FastF1, OpenF1, Jolpica, and curated technical updates.
- Real Jolpica ingestion with deterministic response caching and retries.
- Real FastF1 pilot ingestion for the 2024 British Grand Prix: timing, weather,
  race control, classifications, and clean reference-lap telemetry.
- Cross-fitted 2024 season analysis: each race uses only earlier races for driver,
  constructor, and operational-reliability capability priors.
- Cross-fitted all-season execution from 1950 through 2025.
- Recency-weighted circuit passability from earlier races at the same circuit.
- Real event distance, regulation era, and cached ERA5 weather reanalysis context.
- Status-aware ratings: mechanical/DNS events are excluded, while crashes and
  disqualifications are charged to the driver.
- Interactive driver-rating history plots derived from maintained output files.

The older Hungary fixture remains an engineering acceptance test. The generated 2024
files use real historical source records. The observed target-race finishing result is
retained for evaluation but is not used by that race's features.

## Install and run

Python 3.12+ is required.

```bash
python -m pip install -e ".[dev]"
pytest
xw decompose-event --season 2021 --event "Hungarian Grand Prix" --simulations 20000
xw compare-drivers
xw compare-cars
xw compare-teams
python -m historical_xw.cli run-pilot --season 2024 --event "British Grand Prix"
python -m historical_xw.cli run-season --season 2024
python -m historical_xw.cli run-history --from-season 1950 --to-season 2025
python -m historical_xw.cli rank-drivers --input "data/outputs/race_xw_*.parquet"
python -m historical_xw.cli plot-ratings
```

Use `--no-weather` for a faster offline run. That mode is explicitly labeled with a
neutral missing-weather source in every result rather than pretending the constants are
observed conditions.

The decomposition is written to `data/outputs/race_decompositions.parquet`; `data/outputs/historical_xw.duckdb` exposes it as `race_decompositions`.

## Mathematical contract

For each driver, the model computes a field-normalized coalition value `v(S)` for every subset of `{driver, car, team}`. Missing components are replaced by the configured neutral reference while circuit, realized weather, grid, competitors, race distance, and regulation context stay fixed.

For component `i`, the exact grouped Shapley value is:

```text
phi_i = sum over S excluding i [ |S|! (3-|S|-1)! / 3! ] * [v(S union i) - v(S)]
```

Therefore, up to floating-point tolerance:

```text
XW = XW_neutral_baseline + XdW_probability_contribution
                         + XcW_probability_contribution
                         + XtW_probability_contribution
```

Direct replacement uplift (`XW_full - XW_without_component`) is also exported, but it is not substituted for Shapley because direct effects overlap when interactions exist.

`XP` is the expected mean head-to-head finish score implied by the same Plackett-Luce
strengths. It is also decomposed by exact Shapley attribution:

```text
observed_performance = 1 - (position - 1) / (field_size - 1)
performance_surprise = observed_performance - XP
```

The current driver-rating update combines win conversion, xP performance, and a smaller
teammate-controlled residual. Every term and weight is exposed in `RankingConfig`.

## Historical and causal safeguards

- `cross_fitted` is the default analysis mode: target-race winner labels must be excluded from priors and features; race telemetry may estimate race-specific form.
- `retrospective_descriptive` is explicitly labeled when all historical data are used.
- Finishing position and winner status are outcomes, never same-observation explanatory features.
- Driver accidents map to XdW, technical failures to XcW, operational failures to XtW; unexplained variation remains residual.
- Exact overtakes are not claimed before reliable lap-position data. The all-era circuit
  input is a labeled grid-to-finish movement proxy using only earlier editions, with
  recent editions exponentially overweighted.
- Neutral means the condition-appropriate season grid average, not a vector of arbitrary zero-valued raw features.
- The environment is context, not a fourth contribution group.

## Data path to research-grade estimates

1. Cache completed sessions from FastF1, OpenF1, and Jolpica.
2. Convert source records to Parquet canonical tables: events, entries, laps, telemetry, weather, race control, pits, stints, and results.
3. Reconstruct the circuit from multiple clean laps and align telemetry by distance.
4. Fit partially pooled driver, constructor/car, and operational team effects by season, era, segment class, and environment.
5. Cross-fit target races, then simulate their actual historical fields and conditions.
6. Calculate all eight coalitions, Shapley allocations, uncertainty, comparisons, and sensitivity to neutral definitions.

## Known boundaries

The executed model uses empirical-Bayes shrinkage through lagged expanding priors. xP is
position-aware, but historical gaps to the winner are not yet canonical because official
gap coverage is incomplete across eras. The circuit passability feature is a proxy, not
an exact pass count. Silverstone has real FastF1 telemetry, but the all-season model still
uses official classification, grid, reliability, circuit history, race distance, and
weather reanalysis rather than fitting telemetry microsegments for every event. Roughness,
setup, damage, technical upgrades, radio quality, and some strategy labels require curated
inputs and confidence flags. No live, future, forecast, websocket, betting, or upcoming-
event functionality is included.
