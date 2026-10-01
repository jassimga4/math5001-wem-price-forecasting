import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_design import TARGET, TREE_FEATURES
from scripts.point_models import predict_lgbm, select_lgbm


class PointModelTests(unittest.TestCase):
    def test_zero_price_change_forecasts_persistence(self):
        index = pd.date_range("2025-06-20", "2025-07-10 23:55", freq="5min")
        lag = np.arange(len(index), dtype=float)
        frame = pd.DataFrame(
            {
                "mcp_lag_5min": lag,
                "mcp_lag_30min": lag,
                "mcp_lag_60min": lag,
                "mcp_lag_1d": lag,
                "demand_lag_5min": 1000.0,
                "dpv_lag_5min": 10.0,
                "withdrawal_lag_5min": 1.0,
                "cr_raise_lag_5min": 1.0,
                "cr_lower_lag_5min": 1.0,
                "reg_raise_lag_5min": 1.0,
                "reg_lower_lag_5min": 1.0,
                "rtp_last_complete": 50.0,
                "stem_price": 40.0,
                "stem_imbalance": 0.0,
                "hour": index.hour,
                "dow": index.dayofweek,
                "month": 6,
                TARGET: lag,
            },
            index=index,
        )
        bundle = select_lgbm(
            frame,
            TREE_FEATURES,
            TARGET,
            grid=[{"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20}],
            n_estimators=15,
        )
        self.assertEqual(bundle["objective"], "regression_l1")
        self.assertEqual(bundle["target"], "mcp_minus_lag_5min")
        self.assertLess(bundle["inner_mae"], 1e-6)
        self.assertLess(bundle["persistence_inner_mae"], 1e-6)
        pred = predict_lgbm(bundle, frame, TREE_FEATURES)
        self.assertTrue(np.allclose(pred.to_numpy(), lag))


if __name__ == "__main__":
    unittest.main()
