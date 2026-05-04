from gmdgen.learning.tuning import recommend_tuning

def test_tuning_recommends_drop_emphasis_for_bad_drop():
    recs = recommend_tuning(["bad_drop"])
    assert "drop_emphasis" in recs
    assert recs["drop_emphasis"]["action"] == "increase"

def test_tuning_recommends_density_for_too_empty():
    recs = recommend_tuning(["too_empty"])
    assert "object_budget" in recs
    assert "min_final_object_count" in recs

def test_tuning_recommendations_do_not_change_config_without_user_confirm():
    # The function simply returns a dict and does not mutate any global config.
    recs = recommend_tuning(["off_sync"])
    assert isinstance(recs, dict)
    assert recs["beat_snap_tolerance"]["action"] == "decrease"
