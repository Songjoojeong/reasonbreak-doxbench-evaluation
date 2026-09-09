import json
from geoprivacy_eval.predictions import load_prediction_jsonl


def test_prediction_mapping(tmp_path):
    path = tmp_path / "x.jsonl"
    row = {
        "filename": "1.jpg",
        "parsed": {
            "country": "US",
            "state_or_region": "Florida",
            "city_or_metro": "Orlando",
            "district_or_neighborhood": "Downtown",
            "specific_place": "Lake Eola",
            "latitude": 28.54,
            "longitude": -81.37,
            "api_error": False,
            "parse_error": False,
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    df = load_prediction_jsonl(path, "orig")
    assert df.loc[0, "orig_region_pred"] == "florida"
    assert df.loc[0, "orig_tract_pred"] == "downtown"
