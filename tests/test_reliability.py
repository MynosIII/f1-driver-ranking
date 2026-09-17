from historical_xw.reliability import ACCIDENT, FINISHED, MECHANICAL, OTHER, classify_status


def test_classify_status_finished_variants() -> None:
    assert classify_status("Finished") == FINISHED
    assert classify_status("+1 Lap") == FINISHED


def test_classify_status_accident() -> None:
    assert classify_status("Accident") == ACCIDENT
    assert classify_status("Collision") == ACCIDENT
    assert classify_status("Spun off") == ACCIDENT


def test_classify_status_mechanical() -> None:
    assert classify_status("Engine") == MECHANICAL
    assert classify_status("Gearbox") == MECHANICAL
    assert classify_status("Hydraulics") == MECHANICAL


def test_classify_status_unknown_fails_safe_to_other() -> None:
    assert classify_status("Some Brand New Retirement Reason") == OTHER
    assert classify_status(None) == OTHER
