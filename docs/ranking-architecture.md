# F1 driver ranking architecture

This project joins two lines of work:

1. The legacy `E:\F1` research: multi-opponent Elo, constructor pace and reliability,
   overperformance versus car, teammate dominance, prime form, era normalization, and
   experience stabilization.
2. Historical XW: a cross-fitted, circuit-aware probability engine that separates the
   driver (`XdW`), technical car (`XcW`), and operations team (`XtW`) with exact grouped
   Shapley values.

## First combined model

```mermaid
flowchart LR
    A["Historical results, laps, telemetry, weather"] --> B["Cross-fitted priors"]
    B --> C["Driver capability XdW"]
    B --> D["Car capability XcW"]
    B --> E["Team operations XtW"]
    A --> F["Circuit and race context"]
    C --> G["Eight-coalition race simulation"]
    D --> G
    E --> G
    F --> G
    G --> H["Context-conditioned xW and xP for every entry"]
    G --> I["Shapley attribution and neutral counterfactuals"]
    H --> J["Observed winner and normalized finish"]
    J --> K["xW-adjusted score update"]
    I --> L["Driver, car, and team diagnostics"]
    K --> M["Race-by-race driver rating history"]
    M --> N["Prime, peak, current, and stabilized career rating"]
    L --> N
```

For driver `i` in race `r`:

```text
win_surprise(i,r) = observed_win(i,r) - xW(i,r)
finish_score(i,r) = 1 - (position - 1) / (field_size - 1)
performance_surprise(i,r) = finish_score(i,r) - xP(i,r)

delta(i,r) = 32 * win_surprise
           + 24 * performance_surprise
           +  8 * teammate-controlled surprise
```

This replaces the legacy linear car bonus with the richer xW expectation. A driver who
wins with `xW = 0.80` gains `0.20K`; a driver who wins with `xW = 0.10` gains `0.90K`.
A heavily favored driver who loses is penalized more than an outsider. Since xW and the
winner indicator both sum to one across a race, the unfiltered win residual is zero-sum.
The final driver ledger is not zero-sum when non-driver failures are excluded.

The career view retains the legacy prime/stability idea:

```text
career_raw = 0.75 * mean(best 60 post-race ratings)
           + 0.25 * peak post-race rating

experience = races / (races + 40)
career_rating = 1500 + experience * (career_raw - 1500)
```

All constants are explicit in `RankingConfig`; none are learned from the target race.
The published `score_0_100` is only a presentation scale within the selected population.

Mechanical failures, DNS results, and ambiguous retirements receive no driver update.
Crashes, collisions, and disqualifications receive driver-attributable zero performance.
This deliberately makes the driver ledger non-zero-sum when the car removes an opportunity.

## Legacy research mapped into the maintained package

| Legacy source | Useful idea | Combined destination |
| --- | --- | --- |
| `CarConstructor.py` | Constructor pace plus mechanical reliability | `XcW` and `XtW` capability priors |
| `ELO.py`, `elo2.py`, `ELo3.py` | Dynamic rating, multi-opponent expectation, prime, stabilization | `historical_xw.ranking` |
| `ELo3.py` | Mechanical failures excluded from driver blame | XcW/XtW outcome attribution and data-quality rules |
| `ELo3.py` | Overperformance versus car | xW win surprise and xP performance surprise |
| `ELo3.py` | Teammate dominance | teammate-controlled xP residual |
| `IAf1.py` | PCA exploration of composite metrics | optional sensitivity analysis, not the canonical score |
| `Grafico.py` | Driver rating history visualization | maintained interactive `plot-ratings` output |

The large CSVs under `E:\F1` are generated research artifacts. They are not copied into
Git; the maintained pipeline should reproduce canonical Parquet/CSV outputs from source
records and record the configuration used.

## Guardrails and next modeling stages

- Only information available before the target race may contribute to its xW.
- Full-race `XW`, not a same-race result feature, is the expectation in the update.
- xP measures P2-P20 performance; historical time gaps remain an optional future
  enrichment because complete, comparable gap coverage does not exist across all eras.
- Mechanical, driver-fault, and operational failures need source-backed classification;
  ambiguous retirements should carry uncertainty instead of automatic blame.
- Era comparisons require calibration checks by era and leave-one-season-out backtests.
- Ranking weights should be exposed in sensitivity tables before claiming a definitive
  all-time "best" driver.
