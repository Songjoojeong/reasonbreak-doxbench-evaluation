from geoprivacy_eval.common import haversine_km, normalize_text


def test_normalize_text():
    assert normalize_text("  New   York ") == "new york"
    assert normalize_text("") is None


def test_haversine_zero():
    assert haversine_km(37.0, 127.0, 37.0, 127.0) == 0.0
