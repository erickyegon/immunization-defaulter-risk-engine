from src.labeling.vaccine_schedule import expected_vaccines_by_age


def test_expected_vaccines_by_age():
    vaccines = expected_vaccines_by_age(100)
    assert "has_bcg" in vaccines
    assert "has_penta_3" in vaccines
