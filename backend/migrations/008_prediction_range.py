"""Price-range columns: the 90% band on predictions, and in_band on outcomes.

conf_low_90 / conf_high_90 are served next to the 80% band. in_band records
whether the realised close landed inside [conf_low, conf_high], so the Advisor
can show a live range hit rate instead of only the training-time coverage.
"""

from backend.migrate import add_column_if_missing


def up(cur):
    add_column_if_missing(cur, "predictions", "conf_low_90", "DOUBLE")
    add_column_if_missing(cur, "predictions", "conf_high_90", "DOUBLE")
    add_column_if_missing(cur, "prediction_outcomes", "in_band", "INT")
