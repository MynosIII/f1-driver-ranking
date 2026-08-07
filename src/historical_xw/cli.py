from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .decomposition import decompose_event
from .domain import AnalysisMode, SimulationConfig
from .example import hungary_2021
from .storage import AnalyticalStore
from .real_pipeline import analyze_season, ingest_fastf1_pilot
from .ranking import RankingConfig, build_driver_ranking


def _run_example(args: argparse.Namespace) -> pd.DataFrame:
    entries, context = hungary_2021()
    config = SimulationConfig(simulations=args.simulations, seed=args.seed, mode=AnalysisMode(args.mode))
    rows = decompose_event(entries, context, config)
    store = AnalyticalStore(Path(args.output))
    path = store.write_decompositions(rows)
    frame = pd.DataFrame(rows).sort_values("XW", ascending=False)
    print(frame[["driver", "XW", "XdW_probability_contribution", "XcW_probability_contribution", "XtW_probability_contribution"]].head(args.limit).to_string(index=False))
    print(f"\nParquet: {path}")
    return frame


def _compare(args: argparse.Namespace) -> None:
    store = AnalyticalStore(Path(args.output))
    component = {"drivers": "XdW", "cars": "XcW", "teams": "XtW"}[args.entity]
    group = {"drivers": "driver", "cars": "constructor", "teams": "team"}[args.entity]
    query = f'''SELECT {group}, COUNT(*) races,
      SUM({component}_probability_contribution) total_contribution,
      AVG({component}_probability_contribution) average_per_race
      FROM race_decompositions GROUP BY {group} ORDER BY total_contribution DESC'''
    print(store.query(query).to_string(index=False))


def _run_real_pilot(args: argparse.Namespace) -> None:
    counts = ingest_fastf1_pilot(Path(args.output), args.season, args.event)
    season = analyze_season(args.season, Path(args.output), args.simulations)
    pilot = season.loc[season["event"] == args.event].sort_values("XW", ascending=False)
    target = Path(args.output) / "outputs" / "pilot_2024_british_gp_xw.parquet"
    pilot.to_parquet(target, index=False)
    pilot.to_csv(target.with_suffix(".csv"), index=False)
    print(counts)
    print(pilot[["driver", "XW", "XdW_probability_contribution", "XcW_probability_contribution", "XtW_probability_contribution", "observed_winner"]].head(10).to_string(index=False))


def _rank_drivers(args: argparse.Namespace) -> None:
    source = Path(args.input)
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(source.parent.glob(source.name))
    if not paths:
        raise FileNotFoundError(f"No ranking inputs matched: {args.input}")
    frames = [
        pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        for path in paths
    ]
    ranking, history = build_driver_ranking(
        pd.concat(frames, ignore_index=True),
        RankingConfig(k_factor=args.k_factor, prime_races=args.prime_races),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(output / "driver_ranking.parquet", index=False)
    ranking.to_csv(output / "driver_ranking.csv", index=False)
    history.to_parquet(output / "driver_rating_history.parquet", index=False)
    history.to_csv(output / "driver_rating_history.csv", index=False)
    print(ranking.head(args.limit).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xw", description="Historical-only F1 win attribution")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", default="data/outputs")
    common.add_argument("--simulations", type=int, default=20_000)
    common.add_argument("--seed", type=int, default=20210731)
    common.add_argument("--mode", choices=[m.value for m in AnalysisMode], default=AnalysisMode.CROSS_FITTED.value)
    common.add_argument("--limit", type=int, default=20)
    for name in ("analyze-event", "decompose-event", "example"):
        cmd = sub.add_parser(name, parents=[common])
        cmd.add_argument("--season", type=int, default=2021)
        cmd.add_argument("--event", default="Hungarian Grand Prix")
        cmd.set_defaults(func=_run_example)
    for entity in ("drivers", "cars", "teams"):
        cmd = sub.add_parser(f"compare-{entity}")
        cmd.add_argument("--output", default="data/outputs")
        cmd.set_defaults(func=_compare, entity=entity)
    ingest = sub.add_parser("ingest-history")
    ingest.add_argument("--from-season", type=int, required=True)
    ingest.add_argument("--to-season", type=int, required=True)
    ingest.set_defaults(func=lambda a: print(json.dumps({"status": "connector_ready", "historical_only": True, "from": a.from_season, "to": a.to_season})))
    pilot = sub.add_parser("run-pilot")
    pilot.add_argument("--output", default="data")
    pilot.add_argument("--season", type=int, default=2024)
    pilot.add_argument("--event", default="British Grand Prix")
    pilot.add_argument("--simulations", type=int, default=2000)
    pilot.set_defaults(func=_run_real_pilot)
    season = sub.add_parser("run-season")
    season.add_argument("--output", default="data")
    season.add_argument("--season", type=int, default=2024)
    season.add_argument("--simulations", type=int, default=2000)
    season.add_argument("--no-resume", action="store_true")
    season.set_defaults(func=lambda a: print(analyze_season(a.season, Path(a.output), a.simulations, not a.no_resume).groupby("event")["XW"].sum().describe()))
    history = sub.add_parser("run-history")
    history.add_argument("--output", default="data")
    history.add_argument("--from-season", type=int, default=2018)
    history.add_argument("--to-season", type=int, default=2025)
    history.add_argument("--simulations", type=int, default=2000)
    history.set_defaults(func=lambda a: [analyze_season(year, Path(a.output), a.simulations, True) for year in range(a.from_season, a.to_season + 1)])
    ranking = sub.add_parser("rank-drivers")
    ranking.add_argument("--input", default="data/outputs/race_xw_*.parquet")
    ranking.add_argument("--output", default="data/outputs")
    ranking.add_argument("--k-factor", type=float, default=32.0)
    ranking.add_argument("--prime-races", type=int, default=60)
    ranking.add_argument("--limit", type=int, default=20)
    ranking.set_defaults(func=_rank_drivers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
