"""Pure-function tests for ChartCalibrator.cluster_columns.

No OCR or rendering involved, so these run everywhere -- unlike
test_calibration_integration.py, which needs a Windows OCR language pack.
"""

from __future__ import annotations

from sam_backend.trading.calibration import ChartCalibrator


def _anchor(x: float, y: float = 0.0) -> dict:
    return {"x": x, "y": y, "price": 0.0, "text": ""}


def test_a_single_tight_column_stays_together():
    anchors = [_anchor(100.0), _anchor(102.0), _anchor(98.0), _anchor(101.0)]
    columns = ChartCalibrator.cluster_columns(anchors, tolerance=70.0)
    assert len(columns) == 1
    assert len(columns[0]) == 4


def test_two_columns_well_apart_stay_separate():
    axis = [_anchor(1000.0), _anchor(1005.0), _anchor(995.0)]
    watchlist = [_anchor(200.0), _anchor(205.0), _anchor(195.0)]
    columns = ChartCalibrator.cluster_columns(axis + watchlist, tolerance=70.0)
    assert len(columns) == 2
    assert {len(column) for column in columns} == {3}


def test_a_chain_of_labels_does_not_drift_two_columns_together():
    """Each label sits within tolerance of its neighbour, but the axis (near
    x=1000) and an adjacent panel (near x=850) are genuinely two columns 150px
    apart -- more than the 70px tolerance. A clustering rule that only checks
    the most recently added label lets consecutive small steps chain the two
    together; comparing against the column's running mean must keep them
    apart so unrelated numbers never dilute the real axis fit.
    """
    chain = [_anchor(850.0), _anchor(900.0), _anchor(950.0), _anchor(1000.0)]
    columns = ChartCalibrator.cluster_columns(chain, tolerance=70.0)
    assert len(columns) == 2
    xs = [sorted(item["x"] for item in column) for column in columns]
    assert [850.0, 900.0] in xs
    assert [950.0, 1000.0] in xs
