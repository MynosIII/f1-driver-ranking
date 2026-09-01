from __future__ import annotations

import pandas as pd

from historical_xw.rookie_cli_v7_1 import build_parser, plot_coverage_v7_1


def test_v7_1_defaults_to_plotting_every_driver() -> None:
    args = build_parser().parse_args([])
    assert args.output.endswith("rookie_backcast_v7_1")


def test_plot_coverage_includes_short_careers_and_antonelli() -> None:
    history = pd.DataFrame(
        {
            "driver_id": ["antonelli", "veteran", "veteran", "never_finished"],
            "event_index": [1, 1, 2, 1],
            "status_class": ["finished", "mechanical", "finished", "driver_fault"],
        }
    )
    coverage = plot_coverage_v7_1(history)
    assert coverage["all_eligible_drivers_plotted"]
    assert coverage["plot_driver_count"] == 2
    assert coverage["antonelli_history_rows"] == 1
    assert coverage["antonelli_finished_races"] == 1
    assert coverage["antonelli_plotted"]
    assert coverage["plot_missing_driver_ids"] == ["never_finished"]
