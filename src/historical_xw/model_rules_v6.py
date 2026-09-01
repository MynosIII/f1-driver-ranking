from __future__ import annotations

from dataclasses import dataclass
from math import exp
import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RegulationEraV6:
    start_season: int
    name: str
    reset_type: str


# These are major car-performance resets, not every annual clarification.
# The table is deliberately explicit and reviewable; F1DB has no regulations table.
REGULATION_ERAS_V6: tuple[RegulationEraV6, ...] = (
    RegulationEraV6(1950, "original_world_championship_formula", "initial"),
    RegulationEraV6(1952, "formula_two_world_championship", "engine_formula"),
    RegulationEraV6(1954, "two_point_five_litre_formula", "engine_formula"),
    RegulationEraV6(1961, "one_point_five_litre_formula", "engine_formula"),
    RegulationEraV6(1966, "three_litre_formula", "engine_formula"),
    RegulationEraV6(1983, "flat_bottom_aerodynamics", "aerodynamic_reset"),
    RegulationEraV6(1989, "naturally_aspirated_only", "power_unit_reset"),
    RegulationEraV6(1994, "electronic_driver_aids_ban", "technical_reset"),
    RegulationEraV6(1998, "narrow_track_grooved_tyres", "chassis_tyres_reset"),
    RegulationEraV6(2006, "v8_engine_formula", "power_unit_reset"),
    RegulationEraV6(2009, "major_aero_and_kers_reset", "aerodynamic_reset"),
    RegulationEraV6(2014, "turbo_hybrid_formula", "power_unit_reset"),
    RegulationEraV6(2017, "wide_car_aero_formula", "aerodynamic_reset"),
    RegulationEraV6(2022, "ground_effect_formula", "aerodynamic_reset"),
    RegulationEraV6(2026, "active_aero_new_power_unit_formula", "full_reset"),
)


def regulation_era_v6(season: int) -> RegulationEraV6:
    applicable = [era for era in REGULATION_ERAS_V6 if era.start_season <= int(season)]
    if not applicable:
        raise ValueError(f"Season predates the Formula 1 World Championship: {season}")
    return applicable[-1]


def crosses_regulation_reset(prior_season: int, target_season: int) -> bool:
    if target_season <= prior_season:
        raise ValueError("target_season must be after prior_season")
    if prior_season < REGULATION_ERAS_V6[0].start_season:
        return True
    return regulation_era_v6(prior_season).start_season != regulation_era_v6(
        target_season
    ).start_season


@dataclass(frozen=True, slots=True)
class CarryoverConfigV6:
    """Maximum prior influence before empirical stability shrinkage."""

    base_lag_weights: tuple[float, float, float] = (0.20, 0.10, 0.05)
    share_change_scale: float = 0.08
    minimum_matched_constructors: int = 3


def _transition_stability(
    scores: pd.DataFrame,
    from_season: int,
    to_season: int,
    cfg: CarryoverConfigV6,
) -> tuple[float, int, float, float]:
    left = scores[scores["season"].eq(from_season)][["constructor_id", "score"]].rename(
        columns={"score": "score_from"}
    )
    right = scores[scores["season"].eq(to_season)][["constructor_id", "score"]].rename(
        columns={"score": "score_to"}
    )
    joined = left.merge(right, on="constructor_id", how="inner")
    matched = len(joined)
    if matched < cfg.minimum_matched_constructors:
        return 0.0, matched, float("nan"), float("nan")
    rank_correlation = float(joined["score_from"].corr(joined["score_to"], method="spearman"))
    if not np.isfinite(rank_correlation):
        rank_correlation = 0.0
    median_change = float(np.median(np.abs(joined["score_to"] - joined["score_from"])))
    coverage = matched / max(len(left), len(right), 1)
    stability = max(rank_correlation, 0.0) * exp(-median_change / cfg.share_change_scale) * coverage
    return float(np.clip(stability, 0.0, 1.0)), matched, rank_correlation, median_change


def calculate_carryover_weights_v6(
    constructor_scores: pd.DataFrame,
    config: CarryoverConfigV6 | None = None,
) -> pd.DataFrame:
    """Estimate lag weights from normalized constructor performance stability.

    ``score`` must be comparable across seasons, normally a share or a normalized
    mean rather than raw championship points. A major regulation boundary forces
    every prior weight that crosses it to exactly zero.
    """

    cfg = config or CarryoverConfigV6()
    required = {"season", "constructor_id", "score"}
    if missing := required - set(constructor_scores.columns):
        raise ValueError(f"Constructor scores are missing columns: {sorted(missing)}")
    scores = constructor_scores[list(required)].copy()
    scores["season"] = pd.to_numeric(scores["season"], errors="raise").astype(int)
    scores["score"] = pd.to_numeric(scores["score"], errors="raise").astype(float)
    if scores.duplicated(["season", "constructor_id"]).any():
        raise ValueError("Constructor scores must be unique by season and constructor")

    output: list[dict[str, object]] = []
    seasons = sorted(scores["season"].unique())
    for target in seasons:
        for lag, base_weight in enumerate(cfg.base_lag_weights, start=1):
            prior = int(target) - lag
            reset = crosses_regulation_reset(prior, int(target))
            transition_values: list[float] = []
            matched_values: list[int] = []
            correlations: list[float] = []
            changes: list[float] = []
            if not reset and prior in seasons:
                for year in range(prior + 1, int(target) + 1):
                    stability, matched, correlation, change = _transition_stability(
                        scores, year - 1, year, cfg
                    )
                    transition_values.append(stability)
                    matched_values.append(matched)
                    correlations.append(correlation)
                    changes.append(change)
            empirical = float(np.prod(transition_values)) if transition_values else 0.0
            output.append(
                {
                    "season": int(target),
                    "lag": lag,
                    "prior_season": prior,
                    "regulation_era": regulation_era_v6(int(target)).name,
                    "crosses_regulation_reset": reset,
                    "base_weight": base_weight,
                    "empirical_stability": empirical,
                    "effective_weight": 0.0 if reset else base_weight * empirical,
                    "minimum_matched_constructors": min(matched_values, default=0),
                    "mean_rank_correlation": float(np.nanmean(correlations))
                    if correlations
                    else float("nan"),
                    "mean_median_score_change": float(np.nanmean(changes))
                    if changes
                    else float("nan"),
                }
            )
    return pd.DataFrame(output)


def blend_car_model_priors_v6(
    current_scores: pd.DataFrame,
    prior_constructor_scores: pd.DataFrame,
    carryover_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Blend a season-static chassis score with small constructor-lineage priors."""

    required_current = {"season", "car_model_id", "constructor_id", "current_score"}
    if missing := required_current - set(current_scores.columns):
        raise ValueError(f"Current scores are missing columns: {sorted(missing)}")
    prior = prior_constructor_scores.rename(columns={"score": "prior_score"}).copy()
    rows: list[dict[str, object]] = []
    for item in current_scores.itertuples(index=False):
        season = int(item.season)
        weighted = 0.0
        weight_total = 0.0
        contributions: list[str] = []
        for weight in carryover_weights[carryover_weights["season"].eq(season)].itertuples():
            value = prior[
                prior["season"].eq(weight.prior_season)
                & prior["constructor_id"].astype(str).eq(str(item.constructor_id))
            ]
            if value.empty or float(weight.effective_weight) <= 0.0:
                continue
            contribution = float(value.iloc[0]["prior_score"])
            weighted += float(weight.effective_weight) * contribution
            weight_total += float(weight.effective_weight)
            contributions.append(f"lag{weight.lag}:{weight.effective_weight:.6f}")
        current_weight = 1.0
        blended = (current_weight * float(item.current_score) + weighted) / (
            current_weight + weight_total
        )
        rows.append(
            {
                **item._asdict(),
                "prior_weight_total": weight_total,
                "car_score_v6": blended,
                "prior_contributions": "|".join(contributions),
                "regulation_era": regulation_era_v6(season).name,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class SessionEloConfigV6:
    race_k_factor: float = 18.0
    qualifying_fraction: float = 1.0 / 3.0
    elo_scale: float = 260.0
    maximum_qualifying_delta: float = 9.0

    @property
    def qualifying_k_factor(self) -> float:
        return self.race_k_factor * self.qualifying_fraction


def qualifying_elo_deltas_v6(
    session: pd.DataFrame,
    config: SessionEloConfigV6 | None = None,
) -> pd.Series:
    """Calculate a lower-weight, zero-sum qualifying update.

    The input ``pre_session_strength`` may already include the exact car-model
    expectation. Race retirements are intentionally not consulted: using a later
    race outcome to rewrite qualifying would be future leakage.
    """

    cfg = config or SessionEloConfigV6()
    required = {"qualifying_position", "qualifying_eligible", "pre_session_strength"}
    if missing := required - set(session.columns):
        raise ValueError(f"Qualifying session is missing columns: {sorted(missing)}")
    eligible = session["qualifying_eligible"].astype("boolean").fillna(False).astype(bool) & session[
        "qualifying_position"
    ].notna()
    output = pd.Series(np.nan, index=session.index, dtype=float)
    active = session.loc[eligible].copy()
    if len(active) <= 1:
        return output
    strengths = active["pre_session_strength"].to_numpy(dtype=float)
    differences = (strengths[:, None] - strengths[None, :]) / cfg.elo_scale
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(differences, -30.0, 30.0)))
    np.fill_diagonal(probabilities, np.nan)
    expected = np.nanmean(probabilities, axis=1)
    field_size = len(active)
    observed = 1.0 - (
        active["qualifying_position"].to_numpy(dtype=float) - 1.0
    ) / (field_size - 1.0)
    raw = cfg.qualifying_k_factor * (observed - expected)
    raw -= float(raw.mean())
    raw = np.clip(raw, -cfg.maximum_qualifying_delta, cfg.maximum_qualifying_delta)
    raw -= float(raw.mean())
    output.loc[active.index] = raw
    return output


def clean_position_gain_proxy_v6(race: pd.DataFrame) -> pd.DataFrame:
    """Net position movement among classified finishers only.

    Mechanical retirements, crashes, disqualifications and non-starters are
    removed from both sides of every comparison, so their disappearance is not
    mislabeled as an overtake.
    """

    required = {"grid_position", "finish_position", "status_class_v6"}
    if missing := required - set(race.columns):
        raise ValueError(f"Race is missing columns: {sorted(missing)}")
    result = race.copy()
    eligible = (
        result["status_class_v6"].astype(str).eq("finished")
        & result["grid_position"].notna()
        & result["finish_position"].notna()
        & result["grid_position"].gt(0)
    )
    result["clean_pass_proxy_eligible"] = eligible
    result["clean_grid_rank"] = np.nan
    result["clean_finish_rank"] = np.nan
    result["clean_positions_gained"] = np.nan
    active = result.loc[eligible]
    if active.empty:
        return result
    grid_rank = active["grid_position"].rank(method="first").astype(float)
    finish_rank = active["finish_position"].rank(method="first").astype(float)
    result.loc[active.index, "clean_grid_rank"] = grid_rank
    result.loc[active.index, "clean_finish_rank"] = finish_rank
    result.loc[active.index, "clean_positions_gained"] = grid_rank - finish_rank
    return result


def constructor_season_scores_v6(results: pd.DataFrame) -> pd.DataFrame:
    """Create cross-season comparable constructor shares from eligible finishes."""

    required = {"season", "round", "constructor_id_f1db", "finish_position", "status_class_v6"}
    if missing := required - set(results.columns):
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    frame = results.copy()
    eligible = frame["status_class_v6"].eq("finished") & frame["finish_position"].notna()
    frame = frame.loc[eligible].copy()
    field_size = frame.groupby(["season", "round"])["finish_position"].transform("count")
    frame["performance"] = np.where(
        field_size.gt(1),
        1.0 - (frame["finish_position"] - 1.0) / (field_size - 1.0),
        1.0,
    )
    constructor = (
        frame.groupby(["season", "constructor_id_f1db"], as_index=False)["performance"]
        .mean()
        .rename(columns={"constructor_id_f1db": "constructor_id", "performance": "raw_score"})
    )
    minimum = constructor.groupby("season")["raw_score"].transform("min")
    positive = constructor["raw_score"] - minimum + 1e-6
    constructor["score"] = positive / positive.groupby(constructor["season"]).transform("sum")
    return constructor[["season", "constructor_id", "score", "raw_score"]]
