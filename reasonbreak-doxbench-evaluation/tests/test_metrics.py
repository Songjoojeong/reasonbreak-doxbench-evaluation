import pandas as pd
from geoprivacy_eval.metrics import add_hierarchical_metrics, add_distance_metrics


def test_hierarchical_protection_rate():
    df = pd.DataFrame([
        {"state": "Florida", "city": "Orlando", "address": "Lake Eola Park", "orig_region_pred": "florida", "protected_region_pred": "georgia", "orig_metro_pred": "orlando", "protected_metro_pred": "atlanta", "orig_block_pred": "lake eola park", "protected_block_pred": ""}
    ])
    _, summary = add_hierarchical_metrics(df)
    assert summary["region"]["protection_rate"] == 1.0
    assert summary["metro"]["protection_rate"] == 1.0
    assert summary["block"]["protection_rate"] == 1.0


def test_coordinate_protection_rate():
    df = pd.DataFrame([
        {"latitude": 28.54, "longitude": -81.38, "orig_latitude_pred": 28.5401, "orig_longitude_pred": -81.3801, "protected_latitude_pred": 27.95, "protected_longitude_pred": -82.46}
    ])
    _, summary = add_distance_metrics(df, thresholds_km=(1.0,))
    assert summary["within_1km"]["protection_rate"] == 1.0
